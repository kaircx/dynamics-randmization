import torch
import torch.nn.functional as F
from torch.optim import Adam
from replay_buffer import ReplayBuffer
from actor import Actor
from critic import Critic
import numpy as np
from noise import GaussianNoise

class Agent:
    def __init__(self, state_dim, action_dim, hidden_dim, actor_lr, critic_lr, gamma, tau, buffer_size, batch_size):
        self.actor = Actor(state_dim, action_dim, hidden_dim)
        self.critic = Critic(state_dim, action_dim, hidden_dim)
        self.target_actor = Actor(state_dim, action_dim, hidden_dim)
        self.target_critic = Critic(state_dim, action_dim, hidden_dim)

        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=critic_lr)

        self.replay_buffer = ReplayBuffer(buffer_size)
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        self.num_critic_update_iteration = 0
        self.num_actor_update_iteration = 0
        self.update_iteration = 200

        self.noise = GaussianNoise(action_dim)

    def select_action(self, state, add_noise=True):
        state = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state).detach().numpy()[0]
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1.0, 1.0)

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            print("There is not enough samples in the replay buffer to update the networks.")
            return

        for it in range(self.update_iteration):

            states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

            states = torch.FloatTensor(states)
            actions = torch.FloatTensor(actions)
            rewards = torch.FloatTensor(rewards).unsqueeze(1)
            next_states = torch.FloatTensor(next_states)
            dones = torch.FloatTensor(dones).unsqueeze(1)

            # Update critic
            next_actions = self.target_actor(next_states)
            target_q_v = self.target_critic(next_states, next_actions)
            target_q_values = rewards + self.gamma * dones * target_q_v
            q_values = self.critic(states, actions)
            # import ipdb; ipdb.set_trace()
            critic_loss = F.mse_loss(q_values, target_q_values)
            # print("Critic loss: ", critic_loss)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # print(f"Critic Loss: {critic_loss.item()}")

            # Update actor
            policy_loss = -self.critic(states, self.actor(states)).mean()
            self.actor_optimizer.zero_grad()
            policy_loss.backward()
            self.actor_optimizer.step()

            # print(f"Actor (Policy) Loss: {policy_loss.item()}")

            # Soft update target networks
            for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

            for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

            self.num_actor_update_iteration += 1
            self.num_critic_update_iteration += 1


    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)
