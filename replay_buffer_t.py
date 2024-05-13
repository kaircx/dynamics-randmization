
import torch
import random
from collections import deque

class SamplingSizeError(Exception):
    pass

class Episode:
    def __init__(self, goal, env, max_history_timesteps):
        self._states = []
        self._actions = []
        self._rewards = []
        self._terminal = []
        self._achieved_goals = []

        self._history = None
        self._dim_history_atom = 0
        self._max_history_timesteps = max_history_timesteps

        self._goal = torch.tensor(goal, dtype=torch.float32)
        self._env = env

    def add_step(self, action, obs, reward,done):#, achieved_goal, terminal):
        self._actions.append(action)
        self._states.append(obs)
        self._rewards.append(reward)
        self._terminal.append(done)
        history_atom = torch.cat((torch.tensor(action, dtype=torch.float32), torch.tensor(obs, dtype=torch.float32)))

        if self._history is None:
            # 初めてのステップで_historyと_dim_history_atomを初期化
            self._dim_history_atom = history_atom.shape[0]
            self._history = torch.zeros((self._max_history_timesteps, self._dim_history_atom), dtype=torch.float32)
            self._history[0] = history_atom
        else:
            # _historyが最大長に達していない場合は追加、達している場合は最古のデータを削除して新しいデータを追加
            if self._history.shape[0] < self._max_history_timesteps:
                self._history = torch.cat((self._history, history_atom.unsqueeze(0)))
            else:
                # 最初の行を削除して新しい行を追加
                self._history = torch.cat((self._history[1:], history_atom.unsqueeze(0)))
        

    def get_history(self, t=-1):
        if t == -1:
            return self._history.requires_grad_(True)
        else:
            start = max(0, t - self._max_history_timesteps)
            end = max(0, t)
            if start == end:
                # 全部ゼロのパディングが必要な場合
                return torch.zeros((self._max_history_timesteps, self._dim_history_atom), dtype=torch.float32).requires_grad_(True)
            zeros_pad = torch.zeros((self._max_history_timesteps - (end - start), self._dim_history_atom), dtype=torch.float32)
            padded_history = torch.cat((zeros_pad, self._history[start:end]))
            return padded_history.requires_grad_(True)

    def get_goal(self):
        return torch.tensor(self._goal,requires_grad=True,dtype=torch.float32)

    def get_terminal(self):
        return torch.tensor(self._terminal)

    def get_actions(self):
        return torch.tensor(self._actions,requires_grad=True,dtype=torch.float32)

    def get_states(self):
        return torch.tensor(self._states,requires_grad=True,dtype=torch.float32)

    def get_rewards(self):
        return torch.tensor(self._rewards,requires_grad=True,dtype=torch.float32)

    def get_env(self):
        return self._env

    def get_achieved_goals(self):
        return torch.tensor(self._achieved_goals,requires_grad=True)

class ReplayBuffer:
    def __init__(self, buffer_size, random_seed=0):
        self._buffer_size = buffer_size
        self._buffer = deque()
        self._current_count = 0
        random.seed(random_seed)

    def size(self):
        return self._current_count

    def add(self, episode):
        if self._current_count >= self._buffer_size:
            self._buffer.popleft()
            self._current_count -= 1

        self._buffer.append(episode)
        self._current_count += 1

    def sample_batch(self, batch_size):
        if batch_size > self._current_count:
            raise SamplingSizeError("Sample size greater than number of elements in buffer.")

        return random.sample(list(self._buffer), batch_size)
