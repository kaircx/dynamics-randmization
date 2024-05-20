import argparse
from itertools import count

import os, sys, random
import numpy as np
import pendulum
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from tensorboardX import SummaryWriter

'''
Implementation of Deep Deterministic Policy Gradients (DDPG) with pytorch 
riginal paper: https://arxiv.org/abs/1509.02971
Not the author's implementation !
'''

parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='train', type=str) # mode = 'train' or 'test'
# OpenAI gym environment name, # ['BipedalWalker-v2', 'Pendulum-v0'] or any continuous environment
# Note that DDPG is feasible about hyper-parameters.
# You should fine-tuning if you change to another environment.
parser.add_argument("--env_name", default="CustomInvertedPendulumEnv-v0")
parser.add_argument('--tau',  default=0.005, type=float) # target smoothing coefficient
parser.add_argument('--target_update_interval', default=1, type=int)
parser.add_argument('--test_iteration', default=10, type=int)

parser.add_argument('--learning_rate', default=1e-4, type=float)
parser.add_argument('--gamma', default=0.99, type=int) # discounted factor
parser.add_argument('--capacity', default=1000000, type=int) # replay buffer size
parser.add_argument('--batch_size', default=100, type=int) # mini batch size
parser.add_argument('--seed', default=False, type=bool)
parser.add_argument('--random_seed', default=9527, type=int)
# optional parameters

parser.add_argument('--sample_frequency', default=2000, type=int)
parser.add_argument('--render', default=False, type=bool) # show UI or not
parser.add_argument('--log_interval', default=50, type=int) #
parser.add_argument('--load', default=False, type=bool) # load model
parser.add_argument('--render_interval', default=100, type=int) # after render_interval, the env.render() will work
parser.add_argument('--exploration_noise', default=0.1, type=float)
parser.add_argument('--max_episode', default=100000, type=int) # num of games
parser.add_argument('--print_log', default=5, type=int)
parser.add_argument('--update_iteration', default=200, type=int)
parser.add_argument('--max_length_of_trajectory', default=800, type=int)
parser.add_argument('--history_length', default=1, type=int)
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
script_name = os.path.basename(__file__)
env = gym.make(args.env_name)

if args.seed:
    env.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
param_dim = len(env.get_params())
max_action = float(env.action_space.high[0])
min_Val = torch.tensor(1e-7).float().to(device) # min value

directory = './exp' + script_name + args.env_name +'./'


class Replay_buffer():
    '''
    Code based on:
    https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
    Expects tuples of (state, next_state, action, reward, done)
    '''
    def __init__(self, max_size=args.capacity):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def push(self, data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        x, y, u, r, d, p ,h= [], [], [], [], [], [], []

        for i in ind:
            X, Y, U, R, D, P, H = self.storage[i]
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))
            p.append(np.array(P, copy=False))
            h.append(np.array(H, copy=False))

        return np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1), np.array(p), np.array(h)

UNITS=128
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.max_action = max_action

        self.ff_branch = nn.Sequential(
            nn.Linear(state_dim, UNITS),
            nn.ReLU(),
        )

        self.ff_branch2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, UNITS),
            nn.ReLU(),
        )
        
        self.recurrent_branch = nn.LSTM(input_size=UNITS, hidden_size=UNITS, batch_first=True)
        
        self.merged_branch = nn.Sequential(
            nn.Linear(2 * UNITS, UNITS),
            nn.ReLU(),
            nn.Linear(UNITS, UNITS),
            nn.ReLU(),
            nn.Linear(UNITS, action_dim),
            nn.Tanh()
        )
        
    def forward(self, input_state, history):
        output1 = self.ff_branch(input_state)
        output2 = self.ff_branch2(torch.cat((input_state,history),dim=1))
        hi, (h_n, ff) = self.recurrent_branch(output2.unsqueeze(1))
        merged_input = torch.cat((output1, h_n[-1]),dim=1)
        output = self.merged_branch(merged_input)
        scaled_output = output * self.max_action
        return scaled_output


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.ff_branch = nn.Sequential(
            nn.Linear(state_dim + action_dim + param_dim, UNITS),
            nn.ReLU(),
        )

        self.ff_branch2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, UNITS),
            nn.ReLU(),
        )

        self.recurrent_branch = nn.LSTM(input_size=UNITS, hidden_size=UNITS, batch_first=True)
        
        self.merged_branch = nn.Sequential(
            nn.Linear(2 * UNITS, UNITS),
            nn.ReLU(),
            nn.Linear(UNITS, UNITS),
            nn.ReLU(),
            nn.Linear(UNITS, action_dim),
            nn.Identity()
        )

    def forward(self, param, input_state, input_action, history):
        output1 = self.ff_branch(torch.cat((param, input_action, input_state),dim=1))
        output2 = self.ff_branch2(torch.cat((input_state,history),dim=1))
        hi, (h_n, ff) = self.recurrent_branch(output2.unsqueeze(1))
        merged_input = torch.cat((output1, h_n[-1]),dim=1)
        output = self.merged_branch(merged_input)
        return output


class DDPG(object):
    def __init__(self, state_dim, action_dim, max_action, writer):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)
        self.replay_buffer = Replay_buffer()
        self.writer = writer

        self.num_critic_update_iteration = 0
        self.num_actor_update_iteration = 0
        self.num_training = 0

    def select_action(self, state, history):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        history = torch.FloatTensor(history.reshape(1, -1)).to(device)
        return self.actor(state,history).cpu().data.numpy().flatten()

    def update(self):

        for it in range(args.update_iteration):
            # Sample replay buffer
            x, y, u, r, d, p, h = self.replay_buffer.sample(args.batch_size)
            state = torch.FloatTensor(x).to(device)
            action = torch.FloatTensor(u).to(device)
            next_state = torch.FloatTensor(y).to(device)
            done = torch.FloatTensor(1-d).to(device)
            reward = torch.FloatTensor(r).to(device)
            param = torch.FloatTensor(p).to(device)
            history = torch.FloatTensor(h).to(device)
            # Compute the target Q value
            target_Q = self.critic_target(param, next_state, self.actor_target(next_state,action),action)
            target_Q = reward + (done * args.gamma * target_Q).detach()
            # Get current Q estimate
            current_Q = self.critic(param, state, action, history)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q, target_Q)
            self.writer.add_scalar('Loss/critic_loss', critic_loss, global_step=self.num_critic_update_iteration)
            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Compute actor loss
            actor_loss = -self.critic(param, state, self.actor(state,history), history).mean()
            self.writer.add_scalar('Loss/actor_loss', actor_loss, global_step=self.num_actor_update_iteration)

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            self.num_actor_update_iteration += 1
            self.num_critic_update_iteration += 1

    def save(self):
        torch.save(self.actor.state_dict(), directory + 'actor.pth')
        torch.save(self.critic.state_dict(), directory + 'critic.pth')
        # print("====================================")
        # print("Model has been saved...")
        # print("====================================")

    def load(self):
        self.actor.load_state_dict(torch.load(directory + 'actor.pth'))
        self.critic.load_state_dict(torch.load(directory + 'critic.pth'))
        print("====================================")
        print("model has been loaded...")
        print("====================================")

def main():
    writer = SummaryWriter(directory)
    agent = DDPG(state_dim, action_dim, max_action, writer)
    ep_r = 0
    if args.mode == 'test':
        agent.load()
        for i in range(args.test_iteration):
            state, _ = env.reset()
            history = env.action_space.sample()
            for t in count():
                action = agent.select_action(state,history)
                next_state, reward, done, _, info = env.step(np.float32(action))
                ep_r += reward
                env.render()
                if done or t >= args.max_length_of_trajectory:
                    print("Ep_i \t{}, the ep_r is \t{:0.2f}, the step is \t{}".format(i, ep_r, t))
                    ep_r = 0
                    break
                state = next_state
                history=action

    elif args.mode == 'train':
        if args.load: agent.load()
        total_step = 0
        for i in range(args.max_episode):
            total_reward = 0
            step =0
            state, _= env.reset()
            param = env.get_params()
            history = env.action_space.sample()
            for t in count():
                action = agent.select_action(state, history)
                action = (action + np.random.normal(0, args.exploration_noise, size=env.action_space.shape[0])).clip(
                    env.action_space.low, env.action_space.high)
                next_state, reward, done, _,info = env.step(action)
                if args.render and i >= args.render_interval : env.render()
                agent.replay_buffer.push((state, next_state, action, reward, np.float32(done), param ,history))

                history=action
                state = next_state
                if done or t >= args.max_length_of_trajectory:
                    break
                step += 1
                total_reward += reward
            total_step += step+1
            print("Total T:{} Episode: \t{} Total Reward: \t{:0.2f}".format(total_step, i, total_reward))
            writer.add_scalar('Reward/train', total_reward, i)
            agent.update()
           # "Total T: %d Episode Num: %d Episode T: %d Reward: %f

            if i % args.log_interval == 0:
                agent.save()
    else:
        raise NameError("mode wrong!!!")

if __name__ == '__main__':
    main()