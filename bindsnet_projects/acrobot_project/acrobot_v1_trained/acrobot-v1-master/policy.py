import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

torch.manual_seed(0) # set random seed
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Policy(nn.Module):
    def __init__(self, s_size=6, h_size=32, a_size=3):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(s_size, h_size)
        self.fc2 = nn.Linear(h_size, a_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # print("x: ", x)
        # print("x shape: ", x.shape)
        return F.softmax(x, dim=0)
    
    def act(self, state):
        if isinstance(state, tuple):
            state = torch.from_numpy(state[0]).float().unsqueeze(0).to(device)
        else:
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        # print("act state: ", state)
        probs = self.forward(state).cpu()
        m = Categorical(probs)
        action = m.sample()
        return action.item() - 1, m.log_prob(action)
    
    def save(self, filename):
        torch.save(self.state_dict(), '%s.pth' % (filename))