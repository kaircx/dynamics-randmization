import gymnasium as gym
import numpy as np
import torch
from agent import Agent
import time
def train(agent, env, episodes, max_steps, batch_size):
    for episode in range(episodes):
        state, _ = env.reset()
        env.render()
        # print("Envirnoment reset with state: ", state)
        episode_reward = 0

        for step in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, done, _, _ =  env.step(action * 3)
            # print(action)
            # import ipdb; ipdb.set_trace()

            agent.store_transition(state, action, reward, next_state, done)
            agent.update()

            state = next_state
            episode_reward += reward

            if done:
                # print("Episode finished after {} timesteps".format(step + 1))   
                break
# 
        print(f"Episode {episode + 1}/{episodes}, Reward: {episode_reward}")

if __name__ == "__main__":
    env = gym.make('InvertedPendulum-v4',render_mode='human')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    hidden_dim = 256

    actor_lr = 1e-4
    critic_lr = 1e-3
    gamma = 0.99
    tau = 0.0005
    buffer_size = 100
    batch_size = 64

    agent = Agent(state_dim, action_dim, hidden_dim, actor_lr, critic_lr, gamma, tau, buffer_size, batch_size)
    episodes = 1000
    max_steps = 200

    train(agent, env, episodes, max_steps, batch_size)
