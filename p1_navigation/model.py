import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, modeltype='dqn'):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            modeltype (str): Type of DQN ['dqn', 'double_dqn', 'dueling_dqn']
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.modeltype = modeltype
        
        fc1_units=64
        fc2_units=64
        
        if self.modeltype in ['dqn', 'double_dqn']:
            self.fc1 = nn.Linear(state_size, fc1_units)
            self.fc2 = nn.Linear(fc1_units, fc2_units)
            self.fc3 = nn.Linear(fc2_units, action_size)
            
        elif self.modeltype == 'dueling_dqn':
            self.fc1 = nn.Linear(state_size, fc1_units)
            self.fc2 = nn.Linear(fc1_units, fc2_units)
            
            self.fc_act1 = nn.Linear(fc2_units, fc2_units)
            self.fc_act2 = nn.Linear(fc2_units, action_size)
            
            self.fc_val1 = nn.Linear(fc2_units, fc2_units)
            self.fc_val2 = nn.Linear(fc2_units, 1)
            
        else:
            print('Unknown model type')

    def forward(self, state):
        """Build a network that maps state -> action values."""
        if self.modeltype in ['dqn', 'double_dqn']:
            x = F.relu(self.fc1(state))
            x = F.relu(self.fc2(x))
            return self.fc3(x)
        elif self.modeltype == 'dueling_dqn':
            x = F.relu(self.fc1(state))
            x = F.relu(self.fc2(x))
            
            x_advantage = self.fc_act1(x)
            x_advantage = self.fc_act2(x_advantage)
            
            x_value = self.fc_val1(x)
            x_value = self.fc_val2(x_value)
            
            mean_advantage = x_advantage.mean(1).unsqueeze(1).expand_as(x_advantage)
            x_value = x_value.expand_as(x_advantage)
            
            return x_value + x_advantage - mean_advantage
        else:
            print('Unknown model type')
