import numpy as np
import gymnasium as gym
import itertools
import math
# import pickle
import matplotlib.pyplot as plt

class PlaceCell:
    def __init__(self, rho_rc=1e0, sigma1=np.pi/3, sigma2=np.pi/3,
                 sigma3 = np.arctan(np.pi) / 3,
                 sigma4 = np.arctan(9 * np.pi / 4) / 3, dt=1e-1, n_neurons=1764, **kwargs):
        self.rho_rc = rho_rc
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.sigma3 = sigma3
        self.sigma4 = sigma4
        
        self.dt = dt

        # Create grid indices for m, n, p, q
        self.m_values = np.arange(-2, 4)
        self.n_values = np.arange(-2, 4)
        self.p_values = np.arange(-3, 4)
        self.q_values = np.arange(-3, 4)

        self.input_idx = None
        self.n_neurons = n_neurons

        self.x1 = self.m_values * np.pi / 3
        self.x2 = self.n_values * np.pi / 3
        self.x3 = self.p_values * np.arctan(np.pi) / 3
        self.x4 = self.q_values * np.arctan(9 * np.pi / 4) / 3

    def alpha(self, angle1, angle2):
        diff_angle = (angle2 - angle1) % (2 * np.pi)
        for idx, diff in enumerate(diff_angle):
            if diff > np.pi:
                diff_angle[idx] -= 2 * np.pi
            elif diff < - np.pi:
                diff_angle[idx] += 2 * np.pi

        return diff_angle

    def step(self, state):
        x = state

        theta1 = np.arctan2(x[1], x[0])
        theta2 = np.arctan2(x[3], x[2])

        lambda1 = np.arctan(x[4]/4)
        lambda2 = np.arctan(x[5]/4)

        diff_theta1 = self.alpha(theta1, self.x1)
        diff_theta2 = self.alpha(theta2, self.x2)

        term1 = np.exp(-0.5 * (diff_theta1**2) / (self.sigma1**2))
        term2 = np.exp(-0.5 * (diff_theta2**2) / (self.sigma2**2))
        term3 = np.exp(-0.5 * (lambda1 - self.x3)**2 / (self.sigma3**2))
        term4 = np.exp(-0.5 * (lambda2 - self.x4)**2 / (self.sigma4**2))

        rates = self.rho_rc * \
            (np.multiply.outer(np.multiply.outer(
                np.multiply.outer(term1, term2), term3), term4)).flatten()
            
        s_prob = (1 - np.exp(-rates))

        spikes = s_prob > np.random.rand(self.n_neurons)
        
        return spikes


class SRM0Critic:
    def __init__(self, rho_0=1e0, chi=-5e-3, tau_m=20e-3, threshold=16e-3,
                 delta_u=2e-3, min_voltage=0, dt=1e-1, n_neurons=50, **kwargs):
        # super().__init__(**kwargs)
        self.rho_0 = rho_0
        self.chi = chi
        self.tau_m = tau_m
        self.threshold = threshold
        self.delta_u = delta_u
        self.min_voltage = min_voltage
        self.dt = dt
        self.n_neurons = n_neurons
        self.voltage = np.zeros((self.n_neurons))
        self.critic_refr_trace = np.zeros((self.n_neurons))

    def step(self, J):
        self.critic_refr_trace *= np.exp(-self.dt/tau_m)
        self.voltage += self.chi * self.critic_refr_trace
        
        clipped_J = np.clip(J, -0.01, 0.01)
        self.voltage += clipped_J
        self.voltage[self.voltage < self.min_voltage] = self.min_voltage
        rates = self.rho_0 * np.exp((self.voltage - self.threshold) / self.delta_u)
        s_prob = 1.0 - np.exp(-rates)
        spikes = s_prob > np.random.rand(s_prob.shape[0])
        self.voltage[spikes] = 0
        
        if spikes.any():
            self.critic_refr_trace[spikes] = 1
        
        return spikes
    
class SRM0Actor:
    def __init__(self, rho_0=1e0, chi=-5e-3, tau_m=20e-3, threshold=16e-3,
                 delta_u=2e-3, min_voltage=0, dt=1e-1, n_neurons=60, **kwargs):
        # super().__init__(**kwargs)
        self.rho_0 = rho_0
        self.chi = chi
        self.tau_m = tau_m
        self.threshold = threshold
        self.delta_u = delta_u
        self.min_voltage = min_voltage
        self.dt = dt
        self.n_neurons = n_neurons
        self.voltage = np.zeros((self.n_neurons))
        self.actor_refr_trace = np.zeros((self.n_neurons))
        self.lateral_weights = np.zeros((self.n_neurons, self.n_neurons))
        
        # neurons_to_actions = np.concatenate(
        #     (np.zeros(20), np.ones(20), np.full(20, 2)))
        
        for k in range(self.n_neurons):
            # pdb.set_trace()
            for k_p in range(self.n_neurons):
                if k == k_p:
                    self.lateral_weights[k, k_p] = 0 
                else:
                    self.lateral_weights[k, k_p] = -0.1 + 5 * self.lateral_f(k, k_p) / self.Z_k_f(k)
                # if k == k_p:
                #     continue
                # elif (neurons_to_actions[k] == neurons_to_actions[k_p]):   
                #     lateral_weight = 2e-6
                #     self.lateral_weights[k, k_p] = lateral_weight
                # elif (neurons_to_actions[k] != neurons_to_actions[k_p]):   
                #     lateral_weight = -1e-6 
                #     self.lateral_weights[k, k_p] = lateral_weight
                    
    def lateral_f(self, k, k_p):
        return np.exp(-((k - k_p)/(lateral_lambda)) ** 2)

    def Z_k_f(self, k):
        sum_k_p = np.sum([lateral_f(k, k_p) if k_p != k else 0 for k_p in range(self.n_neurons)])
        return sum_k_p
    
    def step(self, J):
        self.actor_refr_trace *= np.exp(-self.dt/tau_m)
        self.voltage += self.chi * self.actor_refr_trace
        # clipped_J = np.clip(J, -0.01, 0.01)
        J_syn = self.lateral_weights @ J
        J_sum = J + J_syn
        clipped_J_sum = np.clip(J_sum, -0.01, 0.01)
        self.voltage += clipped_J_sum
        self.voltage[self.voltage < self.min_voltage] = self.min_voltage
        rates = self.rho_0 * np.exp((self.voltage - self.threshold) / self.delta_u)
        s_prob = 1.0 - np.exp(-rates)
        spikes = s_prob > np.random.rand(s_prob.shape[0])
        self.voltage[spikes] = 0
        
        if spikes.any():
            self.actor_refr_trace[spikes] = 1
        
        return spikes

class Master():
    def __init__(self,
                 env_name,
                 n_actor,
                 n_critic,
                 n_place,
                 dt,
                 V_0,
                 v,
                 tau_r,
                 stepSize=1,
                 actor_lr=1e3,
                 critic_lr=1e3):

        # gym
        self.env = gym.make(env_name, render_mode="human")
        # self.env.seed(0)
        # self.action_dim = 3
        # if type(self.env.action_space) == gym.spaces.discrete.Discrete:
        #     self.action_dim = self.env.action_space.n
        #     self.discrete_actions = True
        # else:
        #     self.action_dim = self.env.action_space.shape[0]
        #     self.discrete_actions = False

        # self.actors = [SpikingActor() for i in range(5)]
        self.td_errors = []
        self.values = []

        self.state_dim = 6
        self.F_max = 1.00
        self.actor_directions = 2 * self.F_max * \
            np.arange(n_actor) / n_actor - self.F_max
        self.stepsize = stepSize
        self.dt = dt
        self.state = self.env.reset()
        self.reward = 0
        self.done = False
        self.reward_history = [-5000.0]
        self.totalReward = 0
        self.cos1_lim = [-1, 1]
        self.sin1_lim = [-1, 1]
        self.cos2_lim = [-1, 1]
        self.sin2_lim = [-1, 1]
        self.v1_lim = [-12.567, 12.567]
        self.v2_lim = [-28.274, 28.274]
        self.num_upds = 0
        # self.max_steps = 50
        self.step_num = 0
        self.count = 0

        self.V_0 = V_0
        self.v = v
        self.epsilon = 0.1
        self.tau_r = tau_r
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        
        self.v_plus_trace = np.zeros((n_critic))
        self.v_minus_trace = np.zeros((n_critic))
        self.delta_plus_trace = np.zeros((n_critic))
        self.delta_minus_trace = np.zeros((n_critic))
        self.actor_plus_trace = np.zeros((n_actor))
        self.actor_minus_trace = np.zeros((n_actor))
        
        self.ic_eps_plus_trace = np.zeros((n_critic, n_place))
        self.ic_eps_minus_trace = np.zeros((n_critic, n_place))
        self.ic_kappa_plus_trace = np.zeros((n_critic, n_place))
        self.ic_kappa_minus_trace = np.zeros((n_critic, n_place))
        
        self.ia_eps_plus_trace = np.zeros((n_actor, n_place))
        self.ia_eps_minus_trace = np.zeros((n_actor, n_place))
        self.ia_kappa_plus_trace = np.zeros((n_actor, n_place))
        self.ia_kappa_minus_trace = np.zeros((n_actor, n_place))

        # self.place_critic_weights = np.random.normal(
        #     loc=0.5, scale=0.1, size=(n_critic, n_place))
        # self.place_actor_weights = np.random.normal(
        #     loc=0.5, scale=0.1, size=(n_actor, n_place))
        self.place_critic_weights = np.load('weights_0103_spiking_ac/place_critic_weights.npy')
        self.place_actor_weights = np.load('weights_0103_spiking_ac/place_actor_weights.npy')
        self.w_min = 0
        self.w_max = 3
        
        
        self.place_cells = PlaceCell()
        self.actor_cells = SRM0Actor()
        self.critic_cells = SRM0Critic()

    def pipeline(self, time):
        timesteps = int(time / self.dt)
        for timestep in range(timesteps):
            self.state = self.env.reset()
            if isinstance(self.state, tuple):
                self.state = self.state[0]
            self.done = False
            while not self.done:
                self.input_spikes = self.place_cells.step(self.state)
                self.ic_eps_plus_trace *= np.exp(-self.dt/tau_m)
                self.ic_eps_minus_trace *= np.exp(-self.dt/tau_s)
                self.ia_eps_plus_trace *= np.exp(-self.dt/tau_m)
                self.ia_eps_minus_trace *= np.exp(-self.dt/tau_s)
                if self.input_spikes.any():
                    self.ic_eps_plus_trace[:, self.input_spikes] += eps_0 / (tau_m - tau_s)
                    self.ic_eps_minus_trace[:, self.input_spikes] += eps_0 / (tau_m - tau_s)
                    self.ia_eps_plus_trace[:, self.input_spikes] += eps_0 / (tau_m - tau_s)
                    self.ia_eps_minus_trace[:, self.input_spikes] += eps_0 / (tau_m - tau_s)
                
                J_actor = np.sum(self.place_actor_weights * (self.ia_eps_plus_trace - self.ia_eps_minus_trace), axis=1)
                
                self.actor_spikes = self.actor_cells.step(J_actor)
                self.actor_plus_trace *= np.exp(-self.dt/tau_gamma)
                self.actor_minus_trace *= np.exp(-self.dt/v_gamma)
                if self.actor_spikes.any():
                    self.actor_plus_trace[self.actor_spikes] += 1 / (tau_gamma - v_gamma)
                    self.actor_minus_trace[self.actor_spikes] += 1 / (tau_gamma - v_gamma)
                    
                    self.ia_kappa_plus_trace[self.actor_spikes, :] += 1 / ((tau_k - v_k)*tau_r)
                    self.ia_kappa_minus_trace[self.actor_spikes, :] += 1 / ((tau_k - v_k)*tau_r)
                    
                    self.ia_eps_plus_trace[self.actor_spikes, :] = 0
                    self.ia_eps_minus_trace[self.actor_spikes, :] = 0
                    
                self.action = self.select_action()
                
                if isinstance(self.action, np.ndarray):
                    self.action = self.action[0]
            
                self.new_state, self.reward, self.terminated, self.truncated, _ = self.env.step(
                    self.action)
                self.done = self.terminated or self.truncated
                
                if not self.done:
                    self.reward = -10
                    self.totalReward += self.reward
                else:
                    if self.terminated:
                        self.reward = 100
                    else:
                        self.reward = -10
                    # self.reward = -50 # penalize end of episode
                    self.totalReward += self.reward
                    print("\n reward: ", self.totalReward)
                    self.reward_history.append(self.totalReward)
                    # self.state = self.env.reset()
                    self.totalReward = 0
                
                J_critic = np.sum(self.place_critic_weights * (self.ic_eps_plus_trace - self.ic_eps_minus_trace), axis=1)
                
                self.critic_spikes = self.critic_cells.step(J_critic)
                self.v_plus_trace *= np.exp(-self.dt/tau_k)
                self.v_minus_trace *= np.exp(-self.dt/v_k)
                self.delta_plus_trace *= np.exp(-self.dt/v_k)
                self.delta_minus_trace *= np.exp(-self.dt/tau_k)
                
                self.ic_kappa_plus_trace *= np.exp(-self.dt/tau_k)
                self.ic_kappa_minus_trace *= np.exp(-self.dt/v_k)
                self.ia_kappa_plus_trace *= np.exp(-self.dt/tau_k)
                self.ia_kappa_minus_trace *= np.exp(-self.dt/v_k)
                

                if self.critic_spikes.any():
                    # self.critic_refr_trace[self.critic_spikes] = 1
                    self.v_plus_trace[self.critic_spikes] += 1 / (tau_k - v_k)
                    self.v_minus_trace[self.critic_spikes] += 1 / (tau_k - v_k)
                    self.delta_plus_trace[self.critic_spikes] += (1 + tau_r/v_k) / ((tau_k - v_k) * tau_r)
                    self.delta_minus_trace[self.critic_spikes] += (1 + tau_r/tau_k) / ((tau_k - v_k) * tau_r)
                    
                    self.ic_kappa_plus_trace[self.critic_spikes, :] += 1 / ((tau_k - v_k)*tau_r)
                    self.ic_kappa_minus_trace[self.critic_spikes, :] += 1 / ((tau_k - v_k)*tau_r)
                    
                    self.ic_eps_plus_trace[self.critic_spikes, :] = 0
                    self.ic_eps_minus_trace[self.critic_spikes, :] = 0
                
                self.td_error = self.calc_td_error()
                self.value = self.calc_value()
                self.critic_update()
                self.actor_update()
                self.state = self.new_state
                
                
    def calc_td_error(self):
        y_conv_kappa_deriv = self.delta_plus_trace - self.delta_minus_trace
        td_error = self.v * np.mean(y_conv_kappa_deriv) - self.V_0/self.tau_r + self.reward
        self.td_errors.append(td_error)
        return td_error
        
    def critic_update(self):
        td_error = self.td_error
        outer_conv_kappa = (self.ic_eps_plus_trace - self.ic_eps_minus_trace) * (self.ic_kappa_plus_trace - self.ic_kappa_minus_trace)
        
        self.place_critic_weights += self.critic_lr * td_error * outer_conv_kappa
        update_critic = self.critic_lr * td_error * outer_conv_kappa
        self.place_critic_weights = np.clip(
            self.place_critic_weights, a_min=self.w_min, a_max=self.w_max)

    def actor_update(self):
        td_error = self.td_error
        outer_conv_kappa = (self.ia_eps_plus_trace - self.ia_eps_minus_trace) * (self.ia_kappa_plus_trace - self.ia_kappa_minus_trace)
        
        self.place_actor_weights += self.actor_lr * td_error * outer_conv_kappa
        update_actor = self.actor_lr * td_error * outer_conv_kappa
        self.place_actor_weights = np.clip(
            self.place_actor_weights, a_min=self.w_min, a_max=self.w_max)

    def select_action(self):
        y_conv_gamma = self.actor_plus_trace - self.actor_minus_trace
        if np.sum(y_conv_gamma) != 0:
            action = np.dot(y_conv_gamma, self.actor_directions) / np.sum(y_conv_gamma)
        else:
            action = 0
        return action
        # argmax = np.argmax(y_conv_gamma)
        # actor_spikes = self.actor_cells.step(J)
        
        # try:
        #     # y_gamma = np.where(self.actor_spike_times != 0, gamma_filter(self.actor_spike_times - sim.time), self.actor_spike_times)
        #     y_gamma = gamma_filter(self.actor_spike_times - sim.time)
        # except:
        #     y_gamma = gamma_filter(self.actor_spike_times - t)
        #     # y_gamma = np.where(self.actor_spike_times != 0, gamma_filter(self.actor_spike_times - t), self.actor_spike_times)
        # y_conv_gamma = np.sum(y_gamma, axis=1)
    
        # if (np.argmax(y_conv_gamma) < 20):
        #     return 0
        # elif (np.argmax(y_conv_gamma) >= 20 and np.argmax(y_conv_gamma) < 40):
        #     return 1
        # elif(np.argmax(y_conv_gamma) >= 40):
        #     return 2
        # o_rates = []
        # for k in range(len(self.actors)):
        #     o_spikes = self.actors[k].forward(self.state, self.count)
        #     o_rates.append(o_spikes)
        # o_rates = np.array(o_rates)
        # action_rates = np.zeros(3)
        # for k in range(3):
        #     action_rates[k] = sum(
        #         o_rates[np.where(o_rates[:, k] == 1), k][0])
        # action_index = np.argmax(action_rates)
        # action = self.e_greedy_action(action_index)

        # self.count += 1

        # return action

    def calc_value(self):
        rho_conv_kappa = self.v_plus_trace - self.v_minus_trace
        value = self.v * np.mean(rho_conv_kappa) + self.V_0
        self.values.append(value)
        return value


v = 2e-3
V_0 = -40
# V_0 = -16
# v = 300
tau_r = 4  # reward time constant
v_k = 50e-3
tau_k = 200e-3
eps_0 = 1e-4
tau_m = 20e-3
tau_s = 5e-3
tau_gamma = 50e-3
v_gamma = 20e-3

dt = 1e-1
n_actor = 60
n_critic = 50
n_place = 1764
place_radius = 1.5
actor_radius = 2.1
critic_radius = 1.1

stepSize = 1
actor_lr = 1e3
critic_lr = 1e3

lateral_lambda = 0.5
lateral_inhibition = 0.1

rho_rc_value = 1e0
sigma_values = [np.pi/3, np.pi/3, np.arctan(np.pi) / 3, np.arctan(9 * np.pi / 4) / 3]

rho_0 = 1e0
chi = -5e-3
threshold = 16e-3
delta_u = 2e-3

# actor_lambda = 0.5
# lateral_stability_const = (n_actor/actor_lambda/2)**2

def lateral_f(k, k_p):
    return np.exp(-((k - k_p)/(lateral_lambda)) ** 2)

def Z_k_f(k):
    sum_k_p = np.sum([lateral_f(k, k_p) if k_p != k else 0 for k_p in range(n_actor)])
    return sum_k_p



def kappa_plus_filter(time):
    return (np.exp(time / tau_k)) / (tau_k - v_k)

def kappa_minus_filter(time):
    return (np.exp(time / v_k)) / (tau_k - v_k)

def kappa_filter(time):
    return (np.exp(time / tau_k) - np.exp(time / v_k)) / (tau_k - v_k)

def eps_filter(time):
    return eps_0 * (np.exp(time / tau_m) - np.exp(time / tau_s)) / (tau_m - tau_s)

def kappa_deriv_plus_filter(time):
    return ((1+tau_r/v_k)*np.exp(time/v_k) )/(tau_r*(tau_k-v_k))

def kappa_deriv_minus_filter(time):
    return ((1+tau_r/tau_k)*np.exp(time/tau_k))/(tau_r*(tau_k-v_k))

def kappa_deriv_filter(time):
    return (((1+tau_r/v_k)*np.exp(time/v_k) - (1+tau_r/tau_k)*np.exp(time/tau_k)))/(tau_r*(tau_k-v_k))

def gamma_filter(time):
    return (np.exp(time / tau_gamma) - np.exp(time / v_gamma)) / (tau_gamma - v_gamma)


# env_name = 'Acrobot-v1'
env_name = 'AcrobotContinuous-v1'
# env_name = 'MountainCarContinuous-v0'
# env_name = 'Pendulum-v0'
master = Master(
    env_name=env_name,
    n_actor=n_actor,
    n_critic=n_critic,
    n_place=n_place,
    dt=dt,
    V_0=V_0,
    v=v,
    tau_r=tau_r,
    stepSize=stepSize,
    actor_lr=actor_lr,
    critic_lr=critic_lr
)



time = 5000
master.pipeline(time)