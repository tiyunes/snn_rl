# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 21:27:43 2023

@author: yunes
"""

import sys
import os

import torch
import numpy as np
import matplotlib.pyplot as plt
from bindsnet.environment import GymEnvironment
from bindsnet.learning.reward import AbstractReward, MovingAvgRPE
from bindsnet.network.monitors import Monitor
from bindsnet.analysis.plotting import plot_weights
from interactive_env.agents import CartPoleObserverAgent, ExpertAgent
from interactive_env.pipelines import ToMPipeline

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

def cartpole_observation_encoder(
        datum: torch.Tensor,
        time: int,
        **kwargs,
) -> dict:
    """
    Encode observation vector (Only uses data related to angle). It encodes
    a value and its complement in time. So there are two neurons per value.

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
    angle, velocity = datum[2:4]
    min_angle, max_angle = -0.418, 0.418
    min_velocity, max_velocity = -2, 2

    angle_spikes = _compute_spikes(angle, time, min_angle, max_angle, device)
    velocity_spikes = _compute_spikes(velocity, time, min_velocity,
                                      max_velocity, device)

    spikes = torch.stack([angle_spikes, velocity_spikes]).T

    return {"S2": spikes.unsqueeze(1).byte().to(device)}


def _noise_policy(episode, num_episodes, **kwargs):
    return (1 - episode / num_episodes) ** 2

class CartPoleReward(AbstractReward):
    """
    Computes the error based on whether the episode is done or not. The reward
    amount decreases as the agent observes more good episodes.

    Parameters
    ----------

    Keyword Arguments
    -----------------

    """

    def __init__(self, **kwargs):
        self.alpha = 1.0  # reward multiplier
        self.accumulated_rewards = []  # holder of observation evaluations

    def compute(self, **kwargs):
        """
        Compute the reward.

        Keyword Arguments
        -----------------
        reward : float
            The reward value returned from the environment

        """
        reward = kwargs["reward"]
        return reward * self.alpha if reward > 0 else -self.alpha

    def update(self, **kwargs):
        """
        Update internal attributes.

        Keyword Arguments
        -----------------
        accumulated_reward : float
            The value of accumulated reward in the episode.

        """
        accumulated_reward = kwargs.get("accumulated_reward")
        self.accumulated_rewards.append(accumulated_reward)
        if np.mean(self.accumulated_rewards[-10:]) >= 195:
            self.alpha *= 0.1
            

# Define the environment
environment = GymEnvironment('CartPole-v0')

# Define observer agent
observer = CartPoleObserverAgent(environment, dt=1.0, method='first_spike',
                                 reward_fn=CartPoleReward)

# Define expert agent acting on pretrained weights (weight is multiplied by
# observation vector)
expert = ExpertAgent(environment, method='from_weight',
                     noise_policy=_noise_policy)

# Define the pipeline by which the agents interact
pipeline = ToMPipeline(
    observer_agent=observer,
    expert_agent=expert,
    encoding=cartpole_observation_encoder,
    time=15,
    num_episodes=200,
    representation_time=7,
    log_writer = True,
)

pipeline.observe_learn(weight='./experts/hill_climbing.pt',
                              test_interval=10, num_tests=5)
print("Observation Finished")


plt.plot(pipeline.test_rewards, label='test rewards')
plt.xlabel("episode")
plt.ylabel("reward")
plt.legend()
plt.savefig(f"./plots/test_rewards_{8_11_15_300}.png")

print("mean of test rewards: ", np.mean(pipeline.test_rewards))