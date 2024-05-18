import torch
import torch.nn as nn
import torch.optim as optim

UNITS = 128

class Critic(nn.Module):
    def __init__(self, dim_state, dim_action, dim_env, action_bound, tau, learning_rate):
        super(Critic, self).__init__()
        self.dim_state = dim_state
        self.dim_action = dim_action
        self.dim_env = dim_env
        self.action_bound = action_bound
        self.tau = tau
        self.learning_rate = learning_rate

        # Network Architecture
        self.ff_branch = nn.Sequential(
            nn.Linear(dim_env + dim_action + dim_state, UNITS),
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
            nn.Linear(UNITS, 1)
        )
        
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

    def forward(self, input_env,  input_action, input_state, input_history):
        ff_input = torch.cat((input_env, input_action, input_state), dim=0)
        ff_output = self.ff_branch(ff_input)
        _, (h_n, _) = self.recurrent_branch(input_history)
        h_n = h_n[-1]  # 取最後の隠れ層の状態
        merged_input = torch.cat((ff_output, h_n), dim=0)
        output = self.merged_branch(merged_input)
        return output

    def train(self, input_env, input_state, input_action, input_history, predicted_q_value):
        self.optimizer.zero_grad()
        output = self.forward(input_env, input_action, input_state, input_history)
        loss = nn.MSELoss()(output, predicted_q_value)
        loss.backward()
        self.optimizer.step()
        return output

    def update_target_network(self, target_network):
        for target_param, param in zip(target_network.parameters(), self.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def initialize_target_network(self, target_network):
        target_network.load_state_dict(self.state_dict())

    def action_gradients(self, input_env, input_state, input_action, input_history):
        input_action = input_action.detach().requires_grad_(True)
        self.zero_grad()
        outputs = self.forward(input_env, input_action, input_state, input_history)
        outputs.backward()#must remove
        if input_action.grad is not None:
            return input_action.grad
        else:
            print("No gradient computed for input_action. Check network connectivity and input settings.")
            return None

    def get_num_trainable_vars(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)