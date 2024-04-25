import torch
import torch.nn as nn
import torch.optim as optim

UNITS = 128
MAX_STEPS = 50

class Actor(nn.Module):
    def __init__(self, dim_state, dim_goal, dim_action, action_bound, tau, learning_rate, batch_size):
        super(Actor, self).__init__()
        self.dim_state = dim_state
        self.dim_action = dim_action
        self.dim_goal = dim_goal
        self.action_bound = action_bound
        self.tau = tau
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        # Network Architecture
        self.ff_branch = nn.Sequential(
            nn.Linear(dim_goal + dim_state, UNITS),
            nn.ReLU(),
            nn.Linear(UNITS, UNITS),
            nn.ReLU()
        )
        
        self.recurrent_branch = nn.LSTM(input_size=dim_state + dim_action, hidden_size=UNITS, batch_first=True)
        
        self.merged_branch = nn.Sequential(
            nn.Linear(2 * UNITS, UNITS),
            nn.ReLU(),
            nn.Linear(UNITS, UNITS),
            nn.ReLU(),
            nn.Linear(UNITS, dim_action),
            nn.Tanh()
        )
        
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

    def forward(self, input_state, input_goal, input_memory):
        if len(input_state.shape) == 1:
            input_goal = input_goal.unsqueeze(0) 
            input_state = input_state.unsqueeze(0)
            input_memory = input_memory.unsqueeze(0)
        ff_input = torch.cat((input_goal, input_state), dim=1)
        ff_output = self.ff_branch(ff_input)
        _, (h_n, _) = self.recurrent_branch(input_memory)
        h_n = h_n[-1, :,:]  
        merged_input = torch.cat((ff_output, h_n), dim=1)
        output = self.merged_branch(merged_input)
        scaled_output = output * self.action_bound
        return scaled_output

    def train(self, input_state, input_goal, input_history, a_gradient):
        self.optimizer.zero_grad()
        output = self.forward(input_goal, input_state, input_history)
        loss = -torch.mean(output * a_gradient)
        loss.backward()
        self.optimizer.step()

    def update_target_network(self, target_network):
        for target_param, param in zip(target_network.parameters(), self.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def initialize_target_network(self, target_network):
        target_network.load_state_dict(self.state_dict())
        
    def get_num_trainable_vars(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)