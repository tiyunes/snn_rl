# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 22:46:42 2023

@author: yunes
"""

import numpy as np
import matplotlib.pyplot as plt

# given tc_plus = tc_minus = 7, time = 15
# varying tc_e_trace from 15 to 540 

tc_e_traces = np.array([15, 30, 60, 90, 120, 180, 240, 300, 360, 420, 540])
mean_rewards = np.array([145.47, 145.33, 150.04, 161.55, 149.84, 169.94, 170.97, 
                         176.52, 163.84, 153.4, 90.01])

plt.figure()
plt.plot(tc_e_traces, mean_rewards)
plt.title("Mean test rewards")
plt.xlabel("tc_e_trace")
plt.ylabel("Mean reward")
plt.xticks(tc_e_traces)
plt.show()

#given tc_e_trace = 180, time = 15
# varying tc_plus from to 15 to 120

tc_pluses = np.array([7, 15, 30, 60, 120])
mean_rewards_plus = np.array([169.94, 163.39, 174.59, 146.26, 45.84])

plt.figure()
plt.plot(tc_pluses, mean_rewards_plus)
plt.title("Mean test rewards")
plt.xlabel("tc_plus")
plt.ylabel("Mean reward")
plt.xticks(tc_pluses)
plt.show()

# given tc_plus = 15, time = 15
# varying tc_e_trace from 15 to 720 

tc_e_traces_15 = np.array([15, 22, 30, 60, 120, 180, 240, 300, 360])
mean_rewards_15 = np.array([114.45, 134.24, 125.65, 143.23,
                            147.8, 163.39, 168.35, 158.41, 80.64])

plt.figure()
plt.plot(tc_e_traces_15, mean_rewards_15)
plt.title("Mean test rewards")
plt.xlabel("tc__e_traces")
plt.ylabel("Mean reward")
plt.xticks(tc_e_traces_15)
plt.show()