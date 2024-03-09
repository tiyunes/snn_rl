import numpy as np
import gym
import itertools
import math
import matplotlib.pyplot as plt

class PlaceCell:
    def __init__(self, rho_rc=300, 
                  sigma1=0.0375, sigma2=2.92e-3,
                  dt=2e-3, n_neurons=2401, **kwargs):
        self.rho_rc = rho_rc
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        
        self.dt = dt

        # Create grid indices for m, n, p, q
        self.m_values = np.arange(-24, 25)
        self.n_values = np.arange(-24, 25)

        self.input_idx = None
        self.n_neurons = n_neurons

        self.x1 = self.m_values * self.sigma1 - 0.3
        self.x2 = self.n_values * self.sigma2


    def step(self, state):
        x = state

        x_coord = x[0]
        vel = x[1]

        term1 = np.exp(-0.5 * ((x_coord - self.x1)**2) / (self.sigma1**2))
        term2 = np.exp(-0.5 * ((vel - self.x2)**2) / (self.sigma2**2))

        rates = self.rho_rc * (np.multiply.outer(term1, term2)).flatten()
                
        s_prob = 1 - np.exp(-rates)

        spikes = s_prob > np.random.rand(self.n_neurons)
        
        return spikes


class SRM0:
    def __init__(self, rho_0=1e2, chi=-5e-3, tau_m=20e-3, threshold=16e-3,
                 delta_u=2e-3, min_voltage=0, dt=2e-3, n_neurons=50, **kwargs):
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
        
        clipped_J = np.clip(J, -0.025, 0.025)
        self.voltage += clipped_J
        self.voltage[self.voltage < self.min_voltage] = self.min_voltage
        rates = self.rho_0 * np.exp((self.voltage - self.threshold) / self.delta_u)
        s_prob = 1.0 - np.exp(-rates)
        spikes = s_prob > np.random.rand(s_prob.shape[0])
        self.voltage[spikes] = 0
        
        if spikes.any():
            self.critic_refr_trace[spikes] = 1
        
        return spikes


# class SpikingActor():
#     def __init__(self):
#         self.inputs = 20
#         self.hidden = 5
#         self.outputs = 3
#         self.ih_weights = np.random.rand(self.hidden, self.inputs)
#         self.ih_bias = np.random.rand(self.hidden)
#         self.ho_weights = np.random.rand(self.outputs, self.hidden)
#         self.ho_bias = np.random.rand(self.outputs)
#         self.alpha = 0.1
#         self.h_spikes = np.ones(self.hidden)
#         self.o_spikes = np.ones(self.outputs)
#         self.in_spikes = np.ones(self.inputs)
#         self.hz = np.zeros(self.hidden)
#         self.oz = np.zeros(self.outputs)

#     def input_coding(self, state):
#         maps = list(itertools.combinations(range(int(self.inputs*0.5)), r=5))
#         state_code = -1*np.ones(self.inputs)
#         xb = int(self.inputs*0.5*(state[0] + 1.2)/2.4)
#         vb = int(self.inputs*0.5*(state[1] + 0.07)/0.14) 
#         state_code[list(maps[xb])] = 1
#         state_code[list(np.array((maps[vb])) + int(self.inputs*0.5))] = 1
#         return state_code


#     def forward(self,state,count):
#         inputs = self.input_coding(state)
#         self.in_spikes = inputs

#         z = np.matmul(self.ih_weights, inputs) + self.ih_bias
#         pr = 1/(1 + np.exp(-2*z))
#         self.h_spikes = (pr > np.random.rand(self.hidden)).astype(int)
#         self.h_spikes = 2*self.h_spikes - 1
#         self.hz = np.exp(z) + np.exp(-z)


#         zo = np.matmul(self.ho_weights, self.h_spikes) + self.ho_bias
#         po = 1/(1 + np.exp(-2*zo + 1 ))
#         self.o_spikes = (po > np.random.rand(self.outputs)).astype(int)
#         self.o_spikes = 2*self.o_spikes - 1
#         self.oz = np.exp(zo) + np.exp(-zo)


#         return self.o_spikes

#     def update_weights(self, tderror, state, action):
#         #print(state, action, self.h_spikes, self.o_spikes, tderror)

#         h_grad = self.h_spikes
#         h_grad[np.where(self.h_spikes) == -1] = -2*self.hz
#         h_grad[np.where(self.h_spikes) == 1] = 2*(1-self.hz)
#         self.ih_bias += self.alpha*tderror*h_grad
#         self.ih_weights += self.alpha*tderror*np.outer(h_grad, self.in_spikes)
#         #pdb.set_trace()

#         o_grad = self.o_spikes
#         o_grad[np.where(self.o_spikes) == -1] = -2*self.oz
#         o_grad[np.where(self.o_spikes) == 1] = 2*(1-self.oz)

#         self.ho_bias += self.alpha*tderror*o_grad

#         for i in range(self.outputs):
#             if i == action:
#                 for j in range(self.hidden):
#                     self.ho_weights[i,j] += self.alpha*tderror*o_grad[i]*self.h_spikes[j]
#             if i != action and tderror > 0:
#                 for j in range(self.hidden):
#                     self.ho_weights[i,j] += self.alpha*tderror*o_grad[i]*self.h_spikes[j]





class SpikingActor():
    def __init__(self):
        self.inputs = 2
        self.hidden = 100
        self.outputs = 3
        self.ih_weights = 0.01 * np.random.rand(3, self.hidden, self.inputs)
        self.ih_bias = np.random.rand(self.hidden)
        self.ho_weights = 0.01 * np.random.rand(self.outputs, self.hidden)
        self.ho_bias = np.random.rand(self.outputs)
        self.alpha = 0.001
        self.h_spikes = np.ones(self.hidden)
        self.o_spikes = np.ones(self.outputs)
        self.in_spikes = np.ones(self.inputs)
        self.hz = np.zeros(self.hidden)
        self.oz = np.zeros(self.outputs)

    def forward(self,state,count):
        inputs = state
        self.in_spikes = state

        self.hz = np.zeros((3, self.hidden))
        self.h_spikes = np.ones((3, self.hidden))
        for i in range(3):
            z = np.matmul(self.ih_weights[i], inputs)
            p = 1/(1 + np.exp(-2*z))
            self.h_spikes[i] = (p > np.random.rand(self.hidden)).astype(int)
            self.h_spikes[i] = 2*self.h_spikes[i] - 1
            self.hz[i] = 1 + np.exp(2*z*self.h_spikes[i])


        self.oz = np.zeros(self.outputs)
        self.o_spikes = np.ones(self.outputs)

        for i in range(3):
            zo = np.dot(self.ho_weights[i], self.h_spikes[i])
            po = 1/(1 + np.exp(-2*zo))
            self.o_spikes[i] = (po > np.random.rand(1)).astype(int)
            self.o_spikes[i] = 2*self.o_spikes[i] - 1
            self.oz[i] = 1 + np.exp(2*zo*self.o_spikes[i])

        return self.o_spikes

    def update_weights(self, tderror, state, action, mean_reward):
        if mean_reward > -170 and mean_reward < -130:
            self.alpha = 0.2e-3
        elif mean_reward >= -130:
            self.alpha = 0.005e-2
        else:
            self.alpha = 2e-3

        for i in range(3):
            if i == action:
                self.ih_weights[i] += self.alpha*tderror * \
                    np.outer(2*self.h_spikes[i]/self.hz[i], self.in_spikes)
            else:
                if self.o_spikes[i] == 1:
                    self.ih_weights[i] -= self.alpha*tderror * \
                        np.outer(2*self.h_spikes[i]/self.hz[i], self.in_spikes)
                else:
                    self.ih_weights[i] += self.alpha*tderror * \
                        np.outer(2*self.h_spikes[i]/self.hz[i], self.in_spikes)

        for i in range(3):
            if i == action:
                self.ho_weights[i] += self.alpha*tderror * \
                    np.multiply(2*self.o_spikes[i] /
                                self.oz[i], self.h_spikes[i])
            else:
                if self.o_spikes[i] == 1:
                    self.ho_weights[i] -= self.alpha*tderror * \
                        np.multiply(
                            2*self.o_spikes[i]/self.oz[i], self.h_spikes[i])
                else:
                    self.ho_weights[i] += self.alpha*tderror * \
                        np.multiply(
                            2*self.o_spikes[i]/self.oz[i], self.h_spikes[i])


# master class that performs environment interaction and learning
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
                 critic_lr=5e-1):

        # gym
        self.env = gym.make(env_name, render_mode="human")
        # self.env.seed(0)
        self.action_dim = 3
        self.order = 4

        self.actors = [SpikingActor() for i in range(20)]
        self.td_errors = []
        self.values = []
        self.actual_rewards = []
        self.state = None

        self.state_dim = 2
        self.dt = dt
        self.num_states = 2
        self.state = self.env.reset()
        self.reward = 0
        self.done = False
        self.reward_history = [0.0]
        self.totalReward = 0
        self.combns = np.array(list(itertools.product(range(self.order+1), repeat=self.num_states)))
        self.x_lim = [-1.2,0.5]
        self.v_lim = [-0.07, 0.07]
        self.num_upds = 0
        # self.max_steps = 50
        self.step_num = 0
        self.count = 0

        self.V_0 = V_0
        self.v = v
        self.epsilon = 0.5
        self.tau_r = tau_r
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr

        # self.critic_refr_trace = np.zeros((n_critic))
        
        self.place_critic_weights = np.random.normal(
            loc=0.5, scale=0.1, size=(n_critic, n_place))
        self.w_min = 0
        self.w_max = 3
        
        self.place_cells = PlaceCell()
        self.critic_cells = SRM0()
        
    def pipeline(self, time):
        timesteps = int(time / self.dt)
        for timestep in range(timesteps):
            self.v_plus_trace = np.zeros((n_critic))
            self.v_minus_trace = np.zeros((n_critic))
            self.delta_plus_trace = np.zeros((n_critic))
            self.delta_minus_trace = np.zeros((n_critic))
            self.ic_eps_plus_trace = np.zeros((n_critic, n_place))
            self.ic_eps_minus_trace = np.zeros((n_critic, n_place))
            self.ic_kappa_plus_trace = np.zeros((n_critic, n_place))
            self.ic_kappa_minus_trace = np.zeros((n_critic, n_place))
            
            
            self.state = self.env.reset()
            if isinstance(self.state, tuple):
                self.state = self.state[0]
            self.done = False
            while not self.done:
                self.action = self.select_action()
                
                if isinstance(self.action, np.ndarray):
                    self.action = int(self.action[0])
            
                self.new_state, self.reward, self.terminated, self.truncated, _ = self.env.step(
                    self.action)
                self.done = self.terminated or self.truncated
                actual_reward = 100 *((np.sin(3*self.new_state[0])*0.0025 + 0.5  * (self.new_state[1])**2) -\
                (np.sin(3*self.state[0])*0.0025 + 0.5  * (self.state[1])**2))
                self.actual_rewards.append(actual_reward)
                
                if not self.done:
                    # self.reward = np.cos(self.new_state[2])
                    # self.reward = -1
                    self.reward = actual_reward
                    self.totalReward -= 1
                else:
                    if self.terminated:
                        self.reward = 0.01
                    else:
                        self.reward = actual_reward
                    # self.reward = -50 # penalize end of episode
                    # self.reward = 0
                    self.totalReward -= 1
                    print("\n reward: ", self.totalReward)
                    self.reward_history.append(self.totalReward)
                    # self.state = self.env.reset()
                    self.totalReward = 0
                
                self.input_spikes = self.place_cells.step(self.state)
                self.ic_eps_plus_trace *= np.exp(-self.dt/tau_m)
                self.ic_eps_minus_trace *= np.exp(-self.dt/tau_s)
                if self.input_spikes.any():
                    self.ic_eps_plus_trace[:, self.input_spikes] += eps_0 / (tau_m - tau_s)
                    self.ic_eps_minus_trace[:, self.input_spikes] += eps_0 / (tau_m - tau_s)
                    
                J = np.sum(self.place_critic_weights * (self.ic_eps_plus_trace - self.ic_eps_minus_trace), axis=1)
                
                self.critic_spikes = self.critic_cells.step(J)
                self.v_plus_trace *= np.exp(-self.dt/tau_k)
                self.v_minus_trace *= np.exp(-self.dt/v_k)
                self.delta_plus_trace *= np.exp(-self.dt/v_k)
                self.delta_minus_trace *= np.exp(-self.dt/tau_k)
                
                self.ic_kappa_plus_trace *= np.exp(-self.dt/tau_k)
                self.ic_kappa_minus_trace *= np.exp(-self.dt/v_k)
                

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
                self.critic_update(np.mean(self.reward_history[-10:]))
                self.actor_update()
                self.state = self.new_state
                
    def e_greedy_action(self, action_ind):
        prob = (self.epsilon/3)*np.ones(3)
        prob[action_ind] = (1 - self.epsilon) + (self.epsilon/3)
        e_action = np.random.choice(3, 1, p=prob)
        return int(e_action)


    def sensor(self):
        if isinstance(self.state, tuple):
            self.state = self.state[0]
        return self.state

    def calc_td_error(self):
        y_conv_kappa_deriv = self.delta_plus_trace - self.delta_minus_trace
        td_error = self.v * np.mean(y_conv_kappa_deriv) - self.V_0/self.tau_r + self.reward
        self.td_errors.append(td_error)
        return td_error
        
    def critic_update(self, mean_reward):
        td_error = self.td_error
        outer_conv_kappa = (self.ic_eps_plus_trace - self.ic_eps_minus_trace) * (self.ic_kappa_plus_trace - self.ic_kappa_minus_trace)
        
        if (mean_reward > -170 and mean_reward < -130):
            self.critic_lr = 5e-2
        elif (mean_reward >= -130):
            self.critic_lr = 1e-1
        else:
            self.critic_lr = 1e0
            
        self.place_critic_weights += self.critic_lr * td_error * outer_conv_kappa
        update_critic = self.critic_lr * td_error * outer_conv_kappa
        self.place_critic_weights = np.clip(
            self.place_critic_weights, a_min=self.w_min, a_max=self.w_max)

    def actor_update(self):
        td_error = self.td_error

        for k in range(len(self.actors)):
            self.actors[k].update_weights(td_error, self.state, int(
                self.action), np.mean(self.reward_history[-10:]))

    def select_action(self):
        o_rates = []
        for k in range(len(self.actors)):
            o_spikes = self.actors[k].forward(self.state, self.count)
            o_rates.append(o_spikes)
        o_rates = np.array(o_rates)
        action_rates = np.zeros(3)
        for k in range(3):
            action_rates[k] = sum(
                o_rates[np.where(o_rates[:, k] == 1), k][0])
        action_index = np.argmax(action_rates)
        action = self.e_greedy_action(action_index)

        self.count += 1

        return action

    def calc_value(self):
        rho_conv_kappa = self.v_plus_trace - self.v_minus_trace
        value = self.v * np.mean(rho_conv_kappa) + self.V_0
        self.values.append(value)
        return value


v = 1e-5
V_0 = 0
# V_0 = -16
# v = 300
tau_r = 1  # reward time constant
v_k = 50e-3
tau_k = 200e-3
eps_0 = 1e-5
tau_m = 20e-3
tau_s = 5e-3
tau_gamma = 50e-3
v_gamma = 20e-3

dt = 2e-3
n_actor = 60
n_critic = 50
n_place = 2401
place_radius = 1.5
actor_radius = 2.1
critic_radius = 1.1

stepSize = 1
actor_lr = 1e3
critic_lr = 5e-1

lateral_lambda = 0.5
lateral_inhibition = 0.1
# time_window = 10
# time_window_critic = 35


rho_rc_value = 300
sigma_values = [0.0375, 2.92e-3]
# sigma_values = [0.036, 0.06*np.pi]

rho_0 = 1e2
chi = -5e-3
threshold = 16e-3
delta_u = 2e-3

# actor_lambda = 0.5
w_plus = 30
w_minus = -60
# lateral_stability_const = (n_actor/actor_lambda/2)**2

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


env_name = 'MountainCar-v0'
# env_name = 'AcrobotContinuous-v1'
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

time = 35000
master.pipeline(time)

