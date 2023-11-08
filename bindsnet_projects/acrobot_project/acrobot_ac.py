# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 19:02:38 2023

@author: yunes
"""

import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import matplotlib.pyplot as plt
from bindsnet.environment import GymEnvironment
from bindsnet.learning.reward import AbstractReward, MovingAvgRPE
from bindsnet.network.monitors import Monitor
from bindsnet.analysis.plotting import plot_weights
from agents import AcrobotObserverAgent, ExpertAcrobotAgent
from pipelines import AcrobotPipeline

def _compute_spikes(
    datum: torch.Tensor,
    time: int,
    low: float,
    high: float,
    device: str
) -> torch.Tensor:
    times = torch.linspace(low, high, time, device=device)
    spike_times = torch.argmin(torch.abs(datum - times))
    spikes = (np.array(spike_times.to('cpu')).astype(int) ==
              range(0, time)).astype(int)
    reverse_spikes = np.flip(spikes).copy()
    return torch.stack([
        torch.from_numpy(spikes).to(device),
        torch.from_numpy(reverse_spikes).to(device)
    ]).byte()

def acrobot_observation_encoder(
        datum: torch.Tensor,
        time: int,
        **kwargs,
) -> dict:
    """

    Parameters
    ----------
    datum : torch.Tensor
        Observation tensor.
    time : int
        Length of spike train per observation.

    Keyword Arguments
    -----------------

    Returns
    -------
    dict
        The tensor of encoded data per input population.

    """
    device = "cpu" if datum.get_device() < 0 else 'cuda'
    datum = datum.squeeze()
    if (len(datum)==6):
        cos1, sin1, cos2, sin2, vel1, vel2 = datum
    else:
        theta1, theta2, vel1, vel2 = datum
        cos1 = np.cos(theta1)
        sin1 = np.sin(theta1)
        cos2 = np.cos(theta2)
        sin2 = np.sin(theta2)
    
    min_cos1, max_cos1 = -1.0, 1.0
    min_sin1, max_sin1 = -1.0, 1.0
    min_cos2, max_cos2 = -1.0, 1.0
    min_sin2, max_sin2 = -1.0, 1.0
    min_vel1, max_vel1 = -12.57, 12.57
    min_vel2, max_vel2 = -28.27, 28.27

    cos1_spikes = _compute_spikes(cos1, time, min_cos1, max_cos1, device)
    sin1_spikes = _compute_spikes(sin1, time, min_sin1, max_sin1, device)
    cos2_spikes = _compute_spikes(cos2, time, min_cos2, max_cos2, device)
    sin2_spikes = _compute_spikes(sin2, time, min_sin2, max_sin2, device)
    vel1_spikes = _compute_spikes(vel1, time, min_vel1, max_vel1, device)
    vel2_spikes = _compute_spikes(vel2, time, min_vel2, max_vel2, device)

    spikes = torch.stack([cos1_spikes, sin1_spikes, cos2_spikes, sin2_spikes, vel1_spikes, vel2_spikes]).T

    return {"S2": spikes.unsqueeze(1).byte().to(device)}
#     return {"S2": torch.reshape(spikes.unsqueeze(1).byte().to(device), (time, 1, 6, 2))}


def _noise_policy(episode, num_episodes, **kwargs):
    return 0

# Define the environment
environment = GymEnvironment('Acrobot-v1', render_mode='human')

# Define observer agent
observer = AcrobotObserverAgent(environment, dt=1.0, method='first_spike')

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


def expert_forward(state, weight='./acrobot_model.pth', **kwargs):
    model = Policy()
    if isinstance(weight, str):
        weight = torch.load(weight)
    model.load_state_dict(weight)
    
    if (state.shape[0] == 4):
        cos1 = np.cos(state[0])
        sin1 = np.sin(state[0])
        cos2 = np.cos(state[1])
        sin2 = np.sin(state[1])
        vel1 = state[2]
        vel2 = state[3]
    else:
        cos1, sin1, cos2, sin2, vel1, vel2 = state
    
    obs = torch.tensor([cos1, sin1, cos2, sin2, vel1, vel2])

    q = model(obs)
    
    return torch.argmax(q).item() - 1

# Define expert agent acting on pretrained weights (weight is multiplied by
# observation vector)
expert = ExpertAcrobotAgent(environment, method='user-defined',
                     noise_policy=_noise_policy)

# Define the pipeline by which the agents interact
pipeline = AcrobotPipeline(
    observer_agent=observer,
    expert_agent=expert,
    encoding=acrobot_observation_encoder,
    time=10,
    num_episodes=200,
    representation_time=5,
    log_writer = True,
    render_interval=5,
)

pipeline.observe_learn(function=expert_forward, weight='./acrobot_model.pth',
                              test_interval=15, num_tests=7)
print("Observation Finished")






