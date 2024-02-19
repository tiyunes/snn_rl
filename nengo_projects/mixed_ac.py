import numpy as np
import nengo
import gym
import itertools
import math
import matplotlib.pyplot as plt
from nengo.dists import Choice, Distribution, Uniform

class PlaceCell(nengo.neurons.NeuronType):
    state = {
        "voltage": Choice([0]),
        "refractory_time": Choice([0]),
    }

    spiking = True

    def __init__(self, rho_rc=400.0, sigma1=np.pi/3, sigma2=np.pi/3,
                 sigma3=np.arctan(np.pi),
                 sigma4=np.arctan(9 * np.pi / 4), **kwargs):
        super().__init__(**kwargs)
        self.rho_rc = rho_rc
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.sigma3 = sigma3
        self.sigma4 = sigma4

        # Create grid indices for m, n, p, q
        self.m_values = np.arange(1, 7)
        self.n_values = np.arange(1, 7)
        self.p_values = np.arange(-3, 4)
        self.q_values = np.arange(-3, 4)

        self.input_idx = None

        self.n_neurons = len(self.m_values) * len(self.n_values) * \
            len(self.p_values) * len(self.q_values)

        self.x1 = self.m_values * np.pi / 3
        self.x2 = self.n_values * np.pi / 3
        self.x3 = self.p_values * np.arctan(np.pi)
        self.x4 = self.q_values * np.arctan(9 * np.pi / 4)

    def current(self, x, gain, bias):
        return x

    def gain_bias(self, max_rates, intercepts):
        gain = np.ones(1)
        bias = np.zeros(1)

        return gain, bias

    def rates(self, x, gain, bias):

        # rates = np.zeros((self.n_neurons))
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

        return rates

    def alpha(self, angle1, angle2):
        diff_angle = (angle2 - angle1) % (2 * np.pi)
        for idx, diff in enumerate(diff_angle):
            if diff > np.pi:
                diff_angle[idx] -= 2 * np.pi
            elif diff < - np.pi:
                diff_angle[idx] += 2 * np.pi

        return diff_angle

    def max_rates_intercepts(self, gain, bias):
        intercepts = -bias / gain
        max_rates = gain * (1 - intercepts)

        return max_rates, intercepts

    def step(self, dt, J, output, voltage, refractory_time):
        if self.input_idx == None:
            for idx, item in enumerate(list(sim.signals.keys())):
                if item.name == "<Node 'sensor'>.out":
                    self.input_idx = idx
        x = list(sim.signals.values())[self.input_idx]

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

        spikes = (1 - np.exp(-rates * dt)) > np.random.rand(self.n_neurons)
        output[:] = np.float64(spikes) / dt
        if spikes.any():
            master.input_lstimes[spikes] = sim.time
            master.input_lsindeces[spikes] = (master.input_lsindeces[spikes] + 1) % time_window
            
            master.input_spike_times[spikes.nonzero()[0], master.input_lsindeces[spikes].astype('i')] = sim.time
        # for neuron in spikes.nonzero()[0]:
        #     master.input_spike_times[neuron].append(sim.time)


class SRM0(nengo.neurons.NeuronType):
    state = {
        "voltage": Choice([0]),
        "refractory_time": Choice([0]),
    }

    spiking = True

    def __init__(self, rho_0=60.0, chi=-5e-3, tau_m=20e-3, threshold=16e-3,
                 delta_u=2e-3, min_voltage=0, **kwargs):
        super().__init__(**kwargs)
        self.rho_0 = rho_0
        self.chi = chi
        self.tau_m = tau_m
        self.threshold = threshold
        self.delta_u = delta_u
        self.min_voltage = min_voltage

    def current(self, x, gain, bias):
        return x

    def gain_bias(self, max_rates, intercepts):
        gain = np.ones(1)
        bias = np.zeros(1)

        return gain, bias

    def max_rates_intercepts(self, gain, bias):
        intercepts = -bias / gain
        max_rates = gain * (1 - intercepts)

        return max_rates, intercepts

    def rates(self, x, gain, bias, voltage):
        return self.rho_0 * np.exp((voltage - self.threshold) / self.delta_u)

    def step(self, dt, J, output, voltage, refractory_time):
        # x = list(sim.signals.values())[2]
        voltage += np.heaviside(sim.time - master.critic_lstimes, 0) * np.float64(
            master.critic_lstimes > 0) * self.chi * np.exp((master.critic_lstimes - sim.time)/self.tau_m)

        clipped_J = np.clip(J, -0.01, 0.01)
        voltage += clipped_J
        voltage[voltage < self.min_voltage] = self.min_voltage
        rates = self.rho_0 * np.exp((voltage - self.threshold) / self.delta_u)
        s_prob = 1.0 - np.exp(-rates * dt)
        spikes = s_prob > np.random.rand(s_prob.shape[0])
        voltage[spikes] = 0
        output[:] = np.float64(spikes) / dt
        refractory_time += dt * np.float64(refractory_time > 0)
        refractory_time[spikes] = dt

        if spikes.any():
            master.critic_lstimes[spikes] = sim.time
            master.critic_lsindeces[spikes] = (master.critic_lsindeces[spikes] + 1) % time_window
            
            master.critic_spike_times[spikes.nonzero()[0], master.critic_lsindeces[spikes].astype('i')] = sim.time
        
        # master.critic_spike_times[spikes.nonzero(), master.step_num % time_window] = sim.time


class ActorCritic():
    def __init__(self, order, epsilon, step_size, sigma=0.1, num_states=6, radial_sigma=None):
        self.num_states = num_states
        self.epsilon = epsilon
        self.alpha = step_size
        self.sigma = sigma
        #self.cartpole = CartPole()
        self.acrobot = gym.make("Acrobot-v1", render_mode="human")
        self.order = order
        self.lda = 0.5
        # self.w = {}

        # self.w[-1] = 5*np.ones(int(math.pow(order+1, num_states)))
        # self.w[1] = 5*np.ones(int(math.pow(order+1, num_states)))

        self.combns = np.array(
            list(itertools.product(range(order+1), repeat=num_states)))
        self.cos1_lim = [-1, 1]
        self.sin1_lim = [-1, 1]
        self.cos2_lim = [-1, 1]
        self.sin2_lim = [-1, 1]
        self.v1_lim = [-12.567, 12.567]
        self.v2_lim = [-28.274, 28.274]
        self.actors = [SpikingActor() for i in range(5)]

    def e_greedy_action(self, action_ind):
        prob = (self.epsilon/3)*np.ones(3)
        prob[action_ind] = (1 - self.epsilon) + (self.epsilon/3)
        e_action = np.random.choice(3, 1, p=prob)
        return int(e_action)


class SpikingActor():
    def __init__(self):
        self.inputs = 6
        self.hidden = 100
        self.outputs = 3
        self.ih_weights = 0.01*np.random.rand(3, self.hidden, self.inputs)
        self.ih_bias = np.random.rand(self.hidden)
        self.ho_weights = 0.01*np.random.rand(self.outputs, self.hidden)
        self.ho_bias = np.random.rand(self.outputs)
        self.alpha = 0.00005
        self.h_spikes = np.ones(self.hidden)
        self.o_spikes = np.ones(self.outputs)
        self.in_spikes = np.ones(self.inputs)
        self.hz = np.zeros(self.hidden)
        self.oz = np.zeros(self.outputs)

    def forward(self, state, count):
        # inputs = self.input_coding(state)
        inputs = state
        self.in_spikes = inputs

        self.hz = np.zeros((3, self.hidden))
        self.h_spikes = np.ones((3, self.hidden))
        for i in range(3):
            z = np.matmul(self.ih_weights[i], inputs)
            # z = np.clip(z, -10, 10)
            p = 1/(1 + np.exp(-2*z))
            self.h_spikes[i] = (p > np.random.rand(self.hidden)).astype(int)
            self.h_spikes[i] = 2*self.h_spikes[i] - 1
            self.hz[i] = 1 + np.exp(2*z*self.h_spikes[i])

        self.oz = np.zeros(self.outputs)
        self.o_spikes = np.ones(self.outputs)

        for i in range(3):
            zo = np.dot(self.ho_weights[i], self.h_spikes[i])
            # zo = np.clip(zo, -10, 10)
            po = 1/(1 + np.exp(-2*zo))
            self.o_spikes[i] = (po > np.random.rand(1)).astype(int)
            self.o_spikes[i] = 2*self.o_spikes[i] - 1
            self.oz[i] = 1 + np.exp(2*zo*self.o_spikes[i])

        return self.o_spikes

    def update_weights(self, tderror, state, action, mean_reward):
        if mean_reward > -200 and mean_reward < -150:
            self.alpha = 0.000005
        elif mean_reward > -150:
            self.alpha = 0.000001
        else:
            self.alpha = 0.00005

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
                 critic_lr=1e0):

        # gym
        self.env = gym.make(env_name, render_mode="human")
        # self.env.seed(0)
        if type(self.env.action_space) == gym.spaces.discrete.Discrete:
            self.action_dim = self.env.action_space.n
            self.discrete_actions = True
        else:
            self.action_dim = self.env.action_space.shape[0]
            self.discrete_actions = False

        self.actors = [SpikingActor() for i in range(5)]
        self.td_errors = []

        self.state_dim = 6
        self.F_max = 0.75
        self.actor_directions = 2 * self.F_max * \
            np.arange(n_actor) / n_actor - self.F_max
        self.stepsize = stepSize
        self.dt = dt
        self.state = self.env.reset()
        self.reward = 0
        self.done = False
        self.reward_history = [-500.0]
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

        # last spike times of input neurons
        self.input_lstimes = np.zeros((n_place))
        self.input_lsindeces = np.zeros((n_place))
        self.input_spike_times = np.zeros((n_place, time_window))
        # self.input_spike_times = [[] for _ in range(n_place)]
        # last spike times of critic neurons
        self.critic_lstimes = np.zeros((n_critic))
        self.critic_lsindeces = np.zeros((n_critic))
        self.critic_spike_times = np.zeros((n_critic, time_window))
        # self.critic_spike_times = [[] for _ in range(n_critic)]
        # self.place_actor_weights = np.load('place_actor_weights.npy')
        # self.place_critic_weights = np.load('place_critic_weights.npy')
        # self.place_actor_weights = np.random.normal(loc=0.5, scale=0.1, size=(n_actor, n_place))
        self.place_critic_weights = np.random.normal(
            loc=0.5, scale=0.1, size=(n_critic, n_place))
        self.w_min = 0
        self.w_max = 3

        
    def e_greedy_action(self, action_ind):
        prob = (self.epsilon/3)*np.ones(3)
        prob[action_ind] = (1 - self.epsilon) + (self.epsilon/3)
        e_action = np.random.choice(3, 1, p=prob)
        return int(e_action)

    def step(self, t, action):
        if isinstance(action, np.ndarray):
            action = int(action[0])
        self.action = action
        self.step_num += 1
#         self.env.render()
        self.state, self.reward, self.terminated, self.truncated, _ = self.env.step(
            action)
        self.done = self.terminated or self.truncated
        if not self.done:
            # self.reward = -10
            self.totalReward += self.reward
        else:
            # if self.terminated:
            #     self.reward = 50
            # else:
            #     self.reward = -10
            # self.reward = -50 # penalize end of episode
            self.totalReward += self.reward
            print("\n reward: ", self.totalReward)
            self.reward_history.append(self.totalReward)
            self.state = self.env.reset()
            self.totalReward = 0
            # self.step_num = 0

    def sensor(self, t):
        if isinstance(self.state, tuple):
            self.state = self.state[0]
        return self.state

    def calc_td_error(self, t, critic_rates):
        try:
            y_kappa_deriv = np.where(self.critic_spike_times != 0, kappa_deriv_filter(self.critic_spike_times - sim.time), self.critic_spike_times)
        except:
            y_kappa_deriv = np.where(self.critic_spike_times != 0, kappa_deriv_filter(self.critic_spike_times - t), self.critic_spike_times)
        y_conv_kappa_deriv = np.sum(y_kappa_deriv, axis=1)
        td_error = self.v * np.mean(y_conv_kappa_deriv) - self.V_0/self.tau_r + self.reward
        self.td_errors.append(td_error)
        #         print('td', critic_rates)
        return td_error
        
    def outer(self, t, x):
        # X_conv_eps = np.zeros((n_place))
        mask = self.critic_lstimes.reshape(n_critic, 1, 1) < np.tile(self.input_spike_times, (n_critic, 1, 1)) 
        X_posterior = np.where(mask, self.input_spike_times, 0)
        X_eps = np.where(X_posterior != 0, eps_filter(X_posterior - sim.time), X_posterior)
        X_conv_eps = np.sum(X_eps, axis=2)
        outer = X_conv_eps.reshape(n_critic, n_place, 1) * self.critic_spike_times.reshape(n_critic, 1, time_window)
        
        outer_kappa = np.where(outer != 0, kappa_filter(outer - sim.time) / tau_r, outer)
        outer_conv_kappa = np.sum(outer_kappa, axis=2)
        # # X_conv_eps = x[:n_place] * self.dt
        # Y = x[n_place:] * self.dt
        
        # outer = np.outer(Y, X_conv_eps).flatten()
        td_error = x[-1]
        
        self.place_critic_weights += self.critic_lr * td_error * outer_conv_kappa
        update_critic = self.critic_lr * td_error * outer_conv_kappa
        self.place_critic_weights = np.clip(
            self.place_critic_weights, a_min=self.w_min, a_max=self.w_max)

        output_return = np.sum(self.place_critic_weights * X_conv_eps, axis=1)

        return np.sum(self.place_critic_weights * X_conv_eps, axis=1)

    def actor(self, t, x):
        td_error = x[-1]

        for k in range(len(self.actors)):
            self.actors[k].update_weights(td_error, self.state, int(
                self.action), np.mean(self.reward_history[-10:]))

        return np.ones((n_actor))

    # def critic(self, t, x):
    #     dVdw = x[:n_place * n_critic].reshape(n_critic, n_place) 
    #     # place_spikes = x[n_place * n_critic:-1] * self.dt
    #     td_error = x[-1]
    #     self.place_critic_weights += self.critic_lr * td_error * dVdw
    #     update_critic = self.critic_lr * td_error * dVdw
    #     self.place_critic_weights = np.clip(
    #         self.place_critic_weights, a_min=self.w_min, a_max=self.w_max)

    #     output_return = self.place_critic_weights @ self.X_conv_eps.T

    #     return self.place_critic_weights @ self.X_conv_eps.T

    def select_action(self, t, actor_rates):
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

    def calc_value(self, t, critic_rates):
        try:
            rho_kappa = np.where(self.critic_spike_times != 0, kappa_filter(self.critic_spike_times - sim.time), self.critic_spike_times)
        except:
            rho_kappa = np.where(self.critic_spike_times != 0, kappa_filter(self.critic_spike_times - t), self.critic_spike_times)
        rho_conv_kappa = np.sum(rho_kappa, axis=1)
        value = self.v * np.mean(rho_conv_kappa) + self.V_0
        return self.v * np.mean(rho_conv_kappa) + self.V_0


v = 2
V_0 = -3.36
# V_0 = -16
# v = 300
tau_r = 4  # reward time constant
v_k = 50e-3
tau_k = 200e-3
eps_0 = 20e-6
tau_m = 20e-3
tau_s = 5e-3
tau_gamma = 50e-3
v_gamma = 20e-3

dt = 1e-5
n_actor = 60
n_critic = 50
n_place = 1764
place_radius = 1.5
actor_radius = 2.1
critic_radius = 1.1

stepSize = 1
actor_lr = 1e3
critic_lr = 1e0

lateral_lambda = 0.5
lateral_inhibition = 0.1
time_window = 10

rho_rc_value = 400.0
sigma_values = [np.pi/3, np.pi/3, np.arctan(np.pi), np.arctan(9 * np.pi / 4)]

rho_0 = 60.0
chi = -5e-3
threshold = 16e-3
delta_u = 2e-3

# actor_lambda = 0.5
w_plus = 30
w_minus = -60
# lateral_stability_const = (n_actor/actor_lambda/2)**2

def kappa_filter(time):
    return (np.exp(time / tau_k) - np.exp(time / v_k)) / (tau_k - v_k)

def eps_filter(time):
    return eps_0 * (np.exp(time / tau_m) - np.exp(time / tau_s)) / (tau_m - tau_s)

def kappa_deriv_filter(time):
    return (((1+tau_r/v_k)*np.exp(time/v_k) - (1+tau_r/tau_k)*np.exp(time/tau_k)))/(tau_r*(tau_k-v_k))


env_name = 'Acrobot-v1'
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

neurons_to_actions = np.concatenate(
    (np.zeros(20), np.ones(20), np.full(20, 2)))

model = nengo.Network(seed=0)
nengo.rc['progress']['progress_bar'] = 'nengo.utils.progress.TerminalProgressBar'

with model:
    state_node = nengo.Node(output=master.sensor, label='sensor')
    place = nengo.Ensemble(
        n_neurons=n_place,
        # 6 dimensions: cos(theta1), sin(theta1), cos(theta2), sin(theta2), omega1, omega2
        dimensions=6,
        neuron_type=PlaceCell(rho_rc=rho_rc_value, sigma1=sigma_values[0], sigma2=sigma_values[1],
                              sigma3=sigma_values[2], sigma4=sigma_values[3]),
        encoders=np.ones((n_place, 6)),
        # scaled_encoders=np.ones((n_place,6)),
        max_rates=rho_rc_value*np.ones((n_place)),
        gain=np.ones((n_place)),
        bias=np.zeros((n_place)))
    nengo.Connection(state_node, place)

    # actor = nengo.Ensemble(
    #     n_neurons=n_actor,
    #     dimensions=n_place,
    #     neuron_type=SRM0(),
    #     # encoders=master.place_actor_weights,
    #     max_rates=rho_0*np.ones((n_actor)),
    #     gain=np.ones((n_actor)),
    #     bias=np.zeros((n_actor)))

    actor_learn = nengo.Node(master.actor,
                             size_in=1,
                             size_out=n_actor)

    action_selection_node = nengo.Node(
        output=master.select_action, size_in=n_actor)

    step_node = nengo.Node(output=master.step, size_in=1)
    nengo.Connection(action_selection_node, step_node, synapse=0)

    #####################################################

    critic = nengo.Ensemble(
        n_neurons=n_critic,
        dimensions=1764,
        neuron_type=SRM0(),
        encoders=master.place_critic_weights,
        max_rates=rho_0*np.ones((n_critic)),
        gain=np.ones((n_critic)),
        bias=np.zeros((n_critic)))

    critic_main = nengo.Node(master.outer,
                              size_in=n_place + n_critic + 1,
                              size_out=n_critic)
    # convolve place spikes with epsilon
    nengo.Connection(place.neurons,
                     critic_main[:n_place],
                     synapse=0)
                     # synapse=tau_m,
                      # transform=((eps_0 * tau_m) / (tau_m - tau_s)))
    # nengo.Connection(place.neurons,
    #                  critic_outer[:n_place],
    #                  synapse=None)
                     # synapse=tau_s,
                     # transform=-((eps_0 * tau_s) / (tau_m - tau_s)))
    # Pass raw critic spikes into critic_outer
    nengo.Connection(
        critic.neurons, critic_main[n_place:-1], synapse=0)

    # critic_learn = nengo.Node(master.critic,
    #                           size_in=n_place * n_critic + 1,
    #                           size_out=n_critic)
    # convolve critic_outer output with k / tau_r
    # nengo.Connection(critic_outer,
    #                  critic_learn[:n_place * n_critic],
    #                  synapse=0)
                     # synapse=tau_k,
                     # transform=tau_k/((tau_k - v_k)*tau_r))
    # nengo.Connection(critic_outer,
    #                  critic_learn[:n_place * n_critic],
    #                  synapse=None)
                     # synapse=v_k,
                     # transform=(-v_k)/((tau_k - v_k)*tau_r))
    # Pass convolved place spikes with epsilon into critic_learn
    # nengo.Connection(place.neurons,
    #                  critic_learn[n_place * n_critic:-1],
    #                  synapse=0)
                     # synapse=tau_m,
                     # transform=((eps_0 * tau_m) / (tau_m - tau_s)))
    # nengo.Connection(place.neurons,
    #                  critic_learn[n_place * n_critic:-1],
    #                  synapse=tau_s,
    #                  transform=-((eps_0 * tau_s) / (tau_m - tau_s)))

    # connect critic_learn to critic neurons
    nengo.Connection(critic_main, critic.neurons, synapse=0)

    td_error_node = nengo.Node(output=master.calc_td_error, size_in=n_critic)
    # convolve critic spikes with (K' - K/tau_r)
    nengo.Connection(critic.neurons,
                     td_error_node,
                     synapse=0)
                     # synapse=nengo.Lowpass(tau=tau_k),
                     # transform=((-tau_r - tau_k)/(tau_r*(tau_k-v_k))))
    # nengo.Connection(critic.neurons,
    #                  td_error_node,
    #                  synapse=nengo.Lowpass(tau=v_k),
    #                  transform=((tau_r + v_k)/(tau_r*(tau_k-v_k))))

    nengo.Connection(td_error_node, actor_learn[-1], synapse=0)
    nengo.Connection(
        td_error_node, critic_main[-1], synapse=0)

    value_node = nengo.Node(output=master.calc_value, size_in=n_critic)
    nengo.Connection(critic.neurons,
                     value_node,
                     synapse=0)
                     # synapse=tau_k,
                     # transform=tau_k/(tau_k - v_k))
    # nengo.Connection(critic.neurons,
    #                  value_node,
    #                  synapse=v_k,
    #                  transform=(-v_k)/(tau_k - v_k))

    # state_probe = nengo.Probe(state_node, synapse=None, sample_every=dt)


sim = nengo.Simulator(model, dt=dt, optimize=False, progress_bar=False)
sim.run(30)
