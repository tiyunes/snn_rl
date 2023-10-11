import gym
from collections import deque
import numpy as np
import time
import torch
torch.manual_seed(0) # set random seed
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from policy import Policy
from gym.wrappers.monitoring.video_recorder import VideoRecorder

env = gym.make('Acrobot-v1')
# env.seed(0)
# print('observation space:', env.observation_space)
# print('action space:', env.action_space)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

policy = Policy().to(device)
optimizer = optim.Adam(policy.parameters(), lr=0.001)

def reinforce(n_episodes=5000, max_t=1000, gamma=1.0, print_every=100):
    scores_deque = deque(maxlen=100)
    scores = []
    for i_episode in range(1, n_episodes+1):
        saved_log_probs = []
        rewards = []
        state = env.reset()
        # print("state: ", state)
        for t in range(max_t):
            action, log_prob = policy.act(state)
            saved_log_probs.append(log_prob)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            rewards.append(reward)
            if done:
                break 
        scores_deque.append(sum(rewards))
        scores.append(sum(rewards))
        
        discounts = [gamma**i for i in range(len(rewards)+1)]
        R = sum([a*b for a,b in zip(discounts, rewards)])
        
        policy_loss = []
        for log_prob in saved_log_probs:
            policy_loss.append(-log_prob * R)
        policy_loss = torch.cat(policy_loss).sum()
        
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()
        
        if i_episode % print_every == 0:
            torch.save(policy.state_dict(), 'checkpoint.pth')
            print('Episode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))        
    return scores

scores = reinforce()
