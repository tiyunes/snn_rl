import os
import torch
import math
import random
import itertools
import pickle
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

from typing import List, Tuple, Dict

from time import time 
from bindsnet.network import load as t
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


from bindsnet.learning import NoOp
from bindsnet.encoding import poisson
from bindsnet.network import load
from bindsnet.network.monitors import Monitor
from bindsnet.models import DiehlAndCook2015v2, Diehl2LayersWTAOutput
from bindsnet.evaluation import assign_labels, update_ngram_scores, all_activity, proportion_weighting, ngram, logreg_predict
from bindsnet.utils import get_square_weights, get_square_assignments
from bindsnet.analysis.plotting import plot_input, plot_spikes, plot_weights, plot_assignments, plot_performance

from scipy.special import softmax

ROOT_DIR = 'C:/Users/yunes/Мой диск/documents/snn_programs/diehl_cartpole/cartpole'

class InputLayer:
    def __init__(self, rho=5000):
        self.num_features = 4
        self.rho = rho
        self.h = np.zeros(self.num_features)
        self.num_points = 15
        
        self.x_values = np.linspace(-2.5, 2.5, self.num_points)
        self.h[0] = self.x_values[1] - self.x_values[0]
        
        self.vel_values = np.linspace(-1.6, 1.6, self.num_points)
        self.h[1] = self.vel_values[1] - self.vel_values[0]
        
        self.theta_values = np.linspace(-0.45, 0.45, self.num_points)
        self.h[2] = self.theta_values[1] - self.theta_values[0]
        
        self.theta_vel_values = np.linspace(-2.6, 2.6, self.num_points)
        self.h[3] = self.theta_vel_values[1] - self.theta_vel_values[0]
        
        self.h = self.h.reshape(self.num_features, 1)

        self.values = np.vstack((self.x_values, self.vel_values, 
                                 self.theta_values, self.theta_vel_values,
                                 ))
        # self.threshold = 0.1 * np.ones((self.num_features, 1))
        
    def step(self, state):
        state = np.array(state).reshape(self.num_features, 1)
        diff = (self.values - state) ** 2
        exp = np.exp(-0.5 * diff / (self.h)**2)
        # prob = softmax(exp, axis=1)
        
        # spikes = (prob > self.threshold).flatten()
        
        return self.rho * torch.from_numpy(exp.flatten()).type(torch.FloatTensor)

def get_scale(mean_reward, ep_reward=0, max_steps=200):
    reward = max(mean_reward, ep_reward)
    if reward == max_steps:
        return 0.0
    else:
        return 100 / reward 

def fourier_feature_state(state, x_lim, v_lim, theta_lim, omega_lim, num_states, combns):
    state_norm = np.zeros(num_states)
    state_norm[0] = (state[0]+x_lim[1])/(x_lim[1]-x_lim[0])
    state_norm[1] = (state[1]+v_lim[1])/(v_lim[1]-v_lim[0])
    state_norm[2] = (state[2]+theta_lim[1])/(theta_lim[1]-theta_lim[0])
    state_norm[3] = (state[3]+omega_lim[1])/(omega_lim[1]-omega_lim[0])

    prod_array = np.array([np.dot(state_norm, i) for i in combns])
    features = np.array(np.cos(np.pi*prod_array))
    return features

def store_actor_state(actor_network):
    state = {
        'X': {
            'x': actor_network.layers['X'].x.clone().detach(),
        },
        'H': {
            'v': actor_network.layers['H'].v.clone().detach(),
            'x': actor_network.layers['H'].x.clone().detach(),
            'refrac_count': actor_network.layers['H'].refrac_count.clone().detach(),
            'decay': actor_network.layers['H'].decay.clone().detach(),
        },
        'Y': {
            'v': actor_network.layers['Y'].v.clone().detach(),
            'x': actor_network.layers['Y'].x.clone().detach(),
            'refrac_count': actor_network.layers['Y'].refrac_count.clone().detach(),
            'decay': actor_network.layers['Y'].decay.clone().detach(),
        },
        'connections': {
            'H_Y': {
                'p_plus': None,
                'p_minus': None,
                'eligibility': None,
                'eligibility_trace': None,
            }
        }
    }

    connection = actor_network.connections[('H', 'Y')].update_rule

    if hasattr(connection, 'p_plus') and connection.p_plus is not None:
        state['connections']['H_Y']['p_plus'] = connection.p_plus.clone().detach()
    if hasattr(connection, 'p_minus') and connection.p_minus is not None:
        state['connections']['H_Y']['p_minus'] = connection.p_minus.clone().detach()
    if hasattr(connection, 'eligibility') and connection.eligibility is not None:
        state['connections']['H_Y']['eligibility'] = connection.eligibility.clone().detach()
    if hasattr(connection, 'eligibility_trace') and connection.eligibility_trace is not None:
        state['connections']['H_Y']['eligibility_trace'] = connection.eligibility_trace.clone().detach()

    return state

def restore_actor_state(actor_network, state):
    actor_network.layers['X'].x.copy_(state['X']['x'])
    
    actor_network.layers['H'].v.copy_(state['H']['v'])
    actor_network.layers['H'].x.copy_(state['H']['x'])
    actor_network.layers['H'].refrac_count.copy_(state['H']['refrac_count'])
    actor_network.layers['H'].decay.copy_(state['H']['decay'])
    
    actor_network.layers['Y'].v.copy_(state['Y']['v'])
    actor_network.layers['Y'].x.copy_(state['Y']['x'])
    actor_network.layers['Y'].refrac_count.copy_(state['Y']['refrac_count'])
    actor_network.layers['Y'].decay.copy_(state['Y']['decay'])
    
    connection = actor_network.connections[('H', 'Y')].update_rule
    if state['connections']['H_Y']['p_plus'] is not None:
        connection.p_plus.copy_(state['connections']['H_Y']['p_plus'])
    if state['connections']['H_Y']['p_minus'] is not None:
        connection.p_minus.copy_(state['connections']['H_Y']['p_minus'])
    if state['connections']['H_Y']['eligibility'] is not None:
        connection.eligibility.copy_(state['connections']['H_Y']['eligibility'])
    if state['connections']['H_Y']['eligibility_trace'] is not None:
        connection.eligibility_trace.copy_(state['connections']['H_Y']['eligibility_trace'])

def e_greedy_action(action_ind, epsilon=0.1):
    prob = (epsilon/2)*np.ones(2)
    prob[action_ind] = (1 - epsilon) + (epsilon/2)
    #e_action = 2*np.random.choice(2,1,p=prob)-1
    pr_array = np.concatenate((np.ones(int(100*prob[1])), np.zeros(int(100*prob[0]))))
    e_action = pr_array[random.randint(0, len(pr_array)-1)]
    return int(e_action)
      
def main(seed=0, 
         n_iter=50, 
         n_neurons_hid=100,
         n_neurons_out=2, 
         inhib_hid=100, 
         inhib_out=100, 
         lr_hid=1e-2, 
         lr_out=1e2,
         lr_decay=1, 
         time=20, 
         dt=0.5,
         theta_plus=0.0, 
         tc_theta_decay=1e7, 
         train=True, 
         network_path=None,
         hidden_layer_path='C:/Users/yunes/Мой диск/documents/snn_programs/diehl_cartpole/cartpole\\params'\
                                '\\cartpole\\diehl_and_cook_2015_wo_norm_upd\\0_100_15600_100.0_0.001_0.99_350_1_0.05_10000000.0_0.5_10_50.pt',
        w_init_hid=0.5, 
        w_init_out=0.5, 
        thresh_hid=-58.0, 
        thresh_out=-62.0,
        refrac_hid=5, 
        refrac_out=5, 
        n_actors=1,
        gamma=0.99,
        one_spike_hid=False,
        one_spike_out=False,
        tc_trace_input=20.0,
        tc_decay_hid=15.0,
        tc_decay_out=15.0,
        tc_trace_hid=10.0,
        tc_trace_out=10.0,
        tc_plus_out=10.0,
        tc_minus_out=10.0,
        tc_e_trace_out=15.0,
        epsilon=0.1,
        ):

    params = [
        seed, n_neurons_hid, n_neurons_out, inhib_hid, inhib_out, lr_hid, lr_out, lr_decay, time, dt, theta_plus,
        tc_theta_decay
    ]

    test_params = [
        seed, n_neurons_hid, n_neurons_out, inhib_hid, inhib_out, lr_hid, lr_out, lr_decay, time, dt,
        theta_plus, tc_theta_decay
    ]

    model_name = '_'.join([str(x) for x in params])

    rewards_arr = []
    td_errors_arr = []
    values_arr = []

    np.random.seed(seed)
    torch.manual_seed(seed)

    n_classes = 2

    # Build network.
    actor_networks_arr = [
            Diehl2LayersWTAOutput(
            n_inpt=60,
            n_neurons_hid=n_neurons_hid,
            n_neurons_out=n_neurons_out,
            inh_hid=inhib_hid,
            inh_out=inhib_out,
            dt=dt,
            nu_hid=[
                0.25 * lr_hid,
                lr_hid
            ],
            nu_out=[
                0.4 * lr_out, 
                lr_out,
            ],
            reduction=None,
            wmin=0.0,
            wmax=1.0,
            norm=None,
            theta_plus=0.0,
            tc_theta_decay=1e7,
            inpt_shape=None,
            hidden_layer_path=hidden_layer_path,
            w_init_hid=w_init_hid,
            w_init_out=w_init_out,
            thresh_hid=thresh_hid,
            thresh_out=thresh_out,
            refrac_hid=refrac_hid,
            refrac_out=refrac_out,
            one_spike_hid=one_spike_hid,
            one_spike_out=one_spike_out,
            tc_trace_input=tc_trace_input,
            tc_decay_hid=tc_decay_hid,
            tc_decay_out=tc_decay_out,
            tc_trace_hid=tc_trace_hid,
            tc_trace_out=tc_trace_out,
            tc_plus_out=tc_plus_out,
            tc_minus_out=tc_minus_out,
            tc_e_trace_out=tc_e_trace_out,
        ) for _ in range(n_actors)
    ]

    spikes_list = []

    for actor_network in actor_networks_arr:
        spikes = {}
        for layer in actor_network.layers: 
            spikes[layer] = Monitor(actor_network.layers[layer], state_vars=['s'], time=time)
            actor_network.add_monitor(spikes[layer], name=f'{layer}_spikes')
        
        spikes_list.append(spikes)

    env = gym.make('CartPole-v0')

    env.reset()

    input_layer = InputLayer()

    running_reward = 10
    lda = 0.5
    alpha = 0.001

    rates = torch.zeros_like(torch.Tensor(n_neurons_hid, n_classes))

    order = 5
    num_states = 4
    combns = np.array(list(itertools.product(range(order+1), repeat=num_states)))
    x_lim = [-3, 3]
    v_lim = [-10, 10]
    theta_lim = [-math.pi/2, math.pi/2]
    omega_lim = [-math.pi, math.pi]

    w_v = np.zeros(int(math.pow(order+1, num_states)))


    for i in range(n_iter):
        state, ep_reward = env.reset(), 0
        reward = 0
        e_v = np.zeros(int(math.pow(order+1, num_states)))
        rt = 1
        gamma = gamma

        for ti in range(1, 10000):  # Don't infinite loop while learning  
            if isinstance(state, tuple):
                state = state[0]

            rates_for_state = input_layer.step(state)
            fourier_state = fourier_feature_state(state, x_lim, v_lim, theta_lim, omega_lim, num_states, combns)
        
            sample = poisson(datum=rates_for_state, time=time, dt=dt)
            inpts = {'X': sample}

            # Run the networks on the input.
            # spike_counts_sum = torch.Tensor([0, 0])
            spike_counts_sum = torch.zeros(2) 
            actor_states = []

            for actor_network, spikes in zip(actor_networks_arr, spikes_list):
                if ti > 1:
                    actor_state = store_actor_state(actor_network)
                    actor_states.append(actor_state)
                
                actor_network.run(inputs=inpts, time=time, reward=0, one_step=False, step=ti-1, learning=False)
                
                if 'Y' in spikes:
                    spike_counts_sum += spikes['Y'].get('s').sum(dim=0)[0]

            prediction = (spike_counts_sum[1] >= spike_counts_sum[0]).long()

            final_prediction = e_greedy_action(prediction, epsilon)

            state, reward, terminated, truncated, _ = env.step(int(final_prediction))
            done = terminated or truncated
            if done:
                reward = 0
            fourier_new_state = fourier_feature_state(state, x_lim, v_lim, theta_lim, omega_lim, num_states, combns)

            e_v = gamma*lda*e_v + fourier_state
            v_s = np.dot(w_v, fourier_state)
            v_ns = np.dot(w_v, fourier_new_state)
            delta_t = reward + gamma*v_ns - v_s
            w_v += alpha*delta_t*e_v

            ep_reward += reward
            for actor_network in actor_networks_arr:
                actor_network.connections[('H', 'Y')].nu = get_scale(running_reward, ep_reward) * lr_out

            # for actor_network in actor_networks_arr:
            #     for c in actor_network.connections:
            #         actor_network.connections[c].update(
            #             mask=None, learning=True, step=ti-1, reward=delta_t,
            #         )

            for actor_network, actor_state in zip(actor_networks_arr, actor_states):
                restore_actor_state(actor_network, actor_state)
                actor_network.run(inputs=inpts, time=time, reward=delta_t, one_step=False, step=ti-1, learning=True)

            td_errors_arr.append(delta_t)
            values_arr.append(v_ns)

            if done:
                break
        
        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
        rewards_arr.append(ep_reward)


        for actor_network in actor_networks_arr:
            actor_network.reset_state_variables()

        print(i, running_reward, ti)



seed = 0
n_iter=7000
n_neurons_hid = 1000
n_neurons_out = 2
inhib_hid = 0.1
inhib_out = 0.1
lr_hid = 1e-5
lr_out = 1e-3
lr_decay = 1 # 0.99
time_ = 20
dt = 0.5
theta_plus = 0.0
tc_theta_decay = 1e7
train=True

w_init_hid = 0.5
w_init_out = 0.5
thresh_hid = -58.0
thresh_out = -64.0

refrac_hid = 5
refrac_out = 0

one_spike_hid=False
one_spike_out=True
tc_trace_input=20.0
tc_decay_hid=15.0
tc_decay_out=15.0
tc_trace_hid=10.0
tc_trace_out=10.0
tc_plus_out=10.0
tc_minus_out=10.0
tc_e_trace_out=15.0


n_actors = 1

model_path = r'C:\Users\yunes\Мой диск\documents\snn_programs\diehl_cartpole\cartpole\cartpole_1000_neurons\params\cartpole\diehl_and_cook_2015_5_epochs_1000_hidden_neurons_60_input_neurons_not_one_spike_refrac_5\\'\
                '0_1000_15600_0.1_1e-05_1.0_20_0.5_0.0_10000000.0_0.5_10_50_preliminary.pt'
auxiliary_path = r'C:\Users\yunes\Мой диск\documents\snn_programs\diehl_cartpole\cartpole\cartpole_1000_neurons\params\cartpole\diehl_and_cook_2015_5_epochs_1000_hidden_neurons_60_input_neurons_not_one_spike_refrac_5\\'\
                'auxiliary_0_1000_15600_0.1_1e-05_1.0_20_0.5_0.0_10000000.0_0.5_10_50_preliminary.pt'


main(
    seed=seed,
    n_iter=n_iter,
    n_neurons_hid=n_neurons_hid,
    n_neurons_out=n_neurons_out, 
    inhib_hid=inhib_hid, 
    inhib_out=inhib_out, 
    lr_hid=lr_hid,
    lr_out=lr_out, 
    lr_decay=lr_decay, 
    time=time_, 
    dt=dt,
    theta_plus=theta_plus, 
    tc_theta_decay=tc_theta_decay, 
    train=train,
    hidden_layer_path=model_path,
    w_init_hid=w_init_hid, 
    w_init_out=w_init_out, 
    thresh_hid=thresh_hid, 
    thresh_out=thresh_out,
    refrac_hid=refrac_hid, 
    refrac_out=refrac_out,
    n_actors=n_actors,
    one_spike_hid=one_spike_hid,
    one_spike_out=one_spike_out,
    tc_trace_input=tc_trace_input,
    tc_decay_hid=tc_decay_hid,
    tc_decay_out=tc_decay_out,
    tc_trace_hid=tc_trace_hid,
    tc_trace_out=tc_trace_out,
    tc_plus_out=tc_plus_out,
    tc_minus_out=tc_minus_out,
    tc_e_trace_out=tc_e_trace_out,
)

print("ok")
