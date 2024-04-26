import torch
import gym
import numpy as np
from actor_t import Actor
from critic_t import Critic
from noise import OrnsteinUhlenbeckActionNoise
from replay_buffer_t import Episode, ReplayBuffer

MAX_STEPS = 50
TAU = 5e-3
LEARNING_RATE = 1e-3
class Agent:
    def __init__(self, env, batch_size):
        self.env = env
        self.batch_size = batch_size

        # Hardcoded dimensions for now
        self.dim_state = 25
        self.dim_goal = 3
        self.dim_action = 4  # Example value
        self.dim_env = 1

        # Actor-Critic Networks
        self.actor = Actor(self.dim_state, self.dim_goal, self.dim_action, env.action_space.high[0], TAU, LEARNING_RATE, batch_size)
        self.critic = Critic(self.dim_state, self.dim_goal, self.dim_action, self.dim_env, env.action_space.high[0], TAU, LEARNING_RATE)#, self.actor.get_num_trainable_vars())

        # Noise Process
        self.action_noise = OrnsteinUhlenbeckActionNoise(mu=torch.zeros(self.dim_action))

        # Target Networks
        self.target_actor = Actor(self.dim_state, self.dim_goal, self.dim_action, env.action_space.high[0], TAU, LEARNING_RATE, batch_size)
        self.target_critic = Critic(self.dim_state, self.dim_goal, self.dim_action, self.dim_env, env.action_space.high[0], TAU, LEARNING_RATE)#, self.actor.get_num_trainable_vars())
        self.actor.initialize_target_network(self.target_actor)
        self.critic.initialize_target_network(self.target_critic)

    def evaluate_actor(self, obs, goal, history):
        return self.actor(obs, goal, history)

    def evaluate_critic(self, obs, action, goal, env, history):
        obs = torch.tensor(obs.reshape(1, self.dim_state), dtype=torch.float32)
        goal = torch.tensor(goal.reshape(1, self.dim_goal), dtype=torch.float32)
        action = torch.tensor(action.reshape(1, self.dim_action), dtype=torch.float32)
        env = torch.tensor(env.reshape(1, self.dim_env), dtype=torch.float32)
        history = torch.tensor(history.reshape(1, -1, self.dim_state + self.dim_action), dtype=torch.float32)
        return self.critic(env, obs, goal, action, history)

    def train_critic(self, obs, action, goal, env, history, predicted_q_value):
        self.critic.train(env, obs, goal, action, history, predicted_q_value)

    def train_actor(self, obs, goal, history, a_gradient):
        self.actor.train(obs, goal, history, a_gradient)

    def update_target_networks(self):
        self.actor.update_target_network(self.target_actor)
        self.critic.update_target_network(self.target_critic)

    def apply_action_noise(self, action):
        noise = self.action_noise()
        return action + noise

    def save_model(self, filename):
        torch.save({'actor_state_dict': self.actor.state_dict(),
                    'critic_state_dict': self.critic.state_dict(),
                    'target_actor_state_dict': self.target_actor.state_dict(),
                    'target_critic_state_dict': self.target_critic.state_dict()}, filename)

    def load_model(self, filename):
        checkpoint = torch.load(filename)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.target_actor.load_state_dict(checkpoint['target_actor_state_dict'])
        self.target_critic.load_state_dict(checkpoint['target_critic_state_dict'])

    def update_networks(self, episodes, gamma):
        for episode in episodes:
            states = episode.get_states()
            actions = episode.get_actions()
            rewards = episode.get_rewards().unsqueeze(1)  # Ensure rewards have the correct shape
            next_states = torch.roll(states, -1, 0)
            envs=episode.get_env()
            goals = torch.unsqueeze(episode.get_goal().clone().detach(), 0).repeat(states.size(0), 1)
            history = torch.stack([episode.get_history(t) for t in range(states.size(0))])
            # Compute target actions and Q-values for the next states
            with torch.no_grad():
                target_actions = self.target_actor(next_states, goals, history)
                envs=torch.from_numpy(envs).to(torch.float32)
                envs=envs.squeeze(0).repeat(goals.size(0),1)
                target_q_values = self.target_critic(envs, goals, target_actions, next_states,  history)
                terminal_tensors = episode.get_terminal().clone().detach()
                if terminal_tensors.dtype != torch.bool:
                    terminal_tensors = terminal_tensors.bool()
                y_i = rewards + gamma * target_q_values * (~terminal_tensors.unsqueeze(1))

            # Train Critic
            self.train_critic(states, actions, goals, envs, history, y_i)

            # Update Actor
            predicted_actions = self.actor(states, goals, history)
            predicted_actions_=predicted_actions.clone().detach()
            predicted_actions_.requires_grad=True
            action_gradients = self.critic.action_gradients(envs, states, goals, predicted_actions_, history)
            self.train_actor(states, goals, history, action_gradients)

        # Soft update target networks
        self.update_target_networks()


    def evaluate_policy(self, randomized_environment, num_rollouts, max_steps):
        success_count = 0
        total_reward = 0
        for _ in range(num_rollouts):
            # Sample the environment and reset it
            randomized_environment.sample_env()
            env=randomized_environment.get_env()
            env_params = randomized_environment.get_params()
            obs_dict, _ = randomized_environment.get_env().reset()
            goal = obs_dict['desired_goal']
            episode = Episode(goal, env_params, max_steps)
            
            # Initialize with the first observation and a random action
            obs = obs_dict['observation']
            achieved = obs_dict['achieved_goal']
            last_action = env.action_space.sample()  # Random initial action
            episode.add_step(last_action, obs, 0, achieved, False)

            truncated = False
            episode_reward = 0
            while not truncated:
                # Get the current state, goal, and history for actor evaluation
                obs = obs_dict['observation']
                history = episode.get_history()
                action = self.evaluate_actor(torch.from_numpy(obs.copy()).type(torch.float32),torch.from_numpy(goal.copy()).type(torch.float32),history) 
                action=action.detach().cpu().numpy()[0]
                # Execute the action and update the episode
                new_obs_dict, step_reward, done, truncated, info = env.step(action)
                new_obs = new_obs_dict['observation']
                achieved = new_obs_dict['achieved_goal']
                reward = env.compute_reward(achieved, goal, {})
                episode_reward += reward
                episode.add_step(action, new_obs, reward, achieved, terminal=done)
                obs_dict = new_obs_dict
            total_reward += episode_reward
            # Check if the episode was successful
            if info.get('is_success', 0.0) > 0.0:
                success_count += 1
            # Close the environment after the test rollout
            randomized_environment.close_env()

        average_reward = total_reward / num_rollouts
        return success_count, average_reward