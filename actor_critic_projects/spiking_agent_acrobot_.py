import numpy as np
import math
import argparse
# import random
import itertools
# import pdb
from matplotlib import pyplot as plt
import pickle
import gym

# Constants

mod_F = 10
m_c = 1
m_p = 0.1
l = 0.5
td = 0.02
g = 9.8


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

    def fourier_feature_state(self, state, method='fourier'):
        state_norm = np.zeros(self.num_states)
        state_norm[0] = (state[0]+self.cos1_lim[1]) / \
            (self.cos1_lim[1]-self.cos1_lim[0])
        state_norm[1] = (state[1]+self.sin1_lim[1]) / \
            (self.sin1_lim[1]-self.sin1_lim[0])
        state_norm[2] = (state[2]+self.cos2_lim[1]) / \
            (self.cos2_lim[1]-self.cos2_lim[0])
        state_norm[3] = (state[3]+self.sin2_lim[1]) / \
            (self.sin2_lim[1]-self.sin2_lim[0])
        state_norm[4] = (state[3]+self.v1_lim[1]) / \
            (self.v1_lim[1]-self.v1_lim[0])
        state_norm[5] = (state[3]+self.v2_lim[1]) / \
            (self.v2_lim[1]-self.v2_lim[0])

        prod_array = np.array([np.dot(state_norm, i) for i in self.combns])
        features = np.array(np.cos(np.pi*prod_array))
        return features

    def e_greedy_action(self, action_ind):
        prob = (self.epsilon/3)*np.ones(3)
        # prob[action_ind] = (1 - self.epsilon) + (self.epsilon/3)
        #e_action = 2*np.random.choice(2,1,p=prob)-1
        # pr_array = np.concatenate((np.ones(int(100*prob[1])), -1*np.ones(int(100*prob[0]))))
        # e_action = pr_array[random.randint(0, len(pr_array)-1)]
        prob[action_ind] = (1 - self.epsilon) + (self.epsilon/3)
        e_action = np.random.choice(3, 1, p=prob)
        return int(e_action)

    # def softmax_selection(self, qvalues, sigma):
    #     eps = 1e-5
    #     qvalues = qvalues + eps
    #     prob = np.exp(sigma*qvalues)/sum(np.exp(sigma*qvalues))
    #     prob[1] = 1-prob[0]
    #     e_action = 2*np.random.choice(2,1,p=prob)-1
    #     return int(e_action)

    def run_actor_critic(self, num_episodes, features='fourier'):
        rewards = []
        #theta = np.random.rand(self.num_states)
        #theta = np.zeros(self.num_states)
        # theta = np.zeros(int(math.pow(self.order+1, self.num_states)))
        w_v = np.zeros(int(math.pow(self.order+1, self.num_states)))
        alpha = 0.001
        for i in range(num_episodes):
            # if i > 500:
            #    self.alpha = 0.001
            #state = np.zeros(4)
            state = self.acrobot.reset()
            # e_theta = np.zeros_like(theta)
            e_v = np.zeros(int(math.pow(self.order+1, self.num_states)))
            rt = -1
            gamma = 1
            count = 0
            # sigma = 1
            if isinstance(state, tuple):
                state = state[0]
            done = False
            while not done:
                # Act using actor
                fourier_state = self.fourier_feature_state(state, features)
                # state_param = np.dot(theta, fourier_state)

                o_rates = []
                for k in range(len(self.actors)):
                    o_spikes = self.actors[k].forward(state, count)
                    o_rates.append(o_spikes)
                o_rates = np.array(o_rates)
                action_rates = np.zeros(3)
                for k in range(3):
                    action_rates[k] = sum(
                        o_rates[np.where(o_rates[:, k] == 1), k][0])
                action_index = np.argmax(action_rates)
                action = self.e_greedy_action(action_index)

                new_state, reward, terminated, truncated, _ = self.acrobot.step(
                    int(action))
                done = terminated or truncated
                fourier_state = self.fourier_feature_state(state, features)
                fourier_new_state = self.fourier_feature_state(
                    new_state, features)

                # Critic update
                e_v = gamma*self.lda*e_v + fourier_state
                v_s = np.dot(w_v, fourier_state)
                v_ns = np.dot(w_v, fourier_new_state)
                delta_t = rt + gamma*v_ns - v_s
                w_v += alpha*delta_t*e_v

                # Actor update

                for k in range(len(self.actors)):
                    self.actors[k].update_weights(delta_t, state, int(action), np.mean(rewards[-10:]))

                if done:
                    break

                state = new_state
                count += 1

            print("Reward after %s episodes: %s" % (i, -count))
            rewards.append(-count)
        return rewards


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

    def input_coding(self, state):
        maps = list(itertools.combinations(range(int(self.inputs*0.167)), r=5))
        state_code = -1*np.ones(self.inputs)
        cos1b = int(self.inputs*0.167*(state[0] + 1)/2)
        sin1b = int(self.inputs*0.167*(state[1] + 1)/2)
        cos2b = int(self.inputs*0.167*(state[2] + 1)/2)
        sin2b = int(self.inputs*0.167*(state[3] + 1)/2)
        v1b = int(self.inputs*0.167*(state[4] + 12.567)/25.134)
        v2b = int(self.inputs*0.167*(state[5] + 28.274)/56.548)
        state_code[list(maps[cos1b])] = 1
        state_code[list(np.array((maps[sin1b])) + int(self.inputs*0.167))] = 1
        state_code[list(np.array((maps[cos2b])) + int(self.inputs*0.334))] = 1
        state_code[list(np.array((maps[sin2b])) + int(self.inputs*0.501))] = 1
        state_code[list(np.array((maps[v1b])) + int(self.inputs*0.668))] = 1
        state_code[list(np.array((maps[v2b])) + int(self.inputs*0.835))] = 1
        return state_code

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

        #############################################################

        # z = np.matmul(self.ih_weights, inputs) + self.ih_bias
        # # z = np.clip(z, -20, 20)
        # pr = 1/(1 + np.exp(-2*z))
        # self.h_spikes = (pr > np.random.rand(self.hidden)).astype(int)
        # self.h_spikes = 2*self.h_spikes - 1
        # self.hz = 1 + np.exp(2*z*self.h_spikes)

        # zo = np.matmul(self.ho_weights, self.h_spikes) + self.ho_bias
        # # zo = np.clip(zo, -20, 20)
        # po = 1/(1 + np.exp(-2*zo + 1))
        # self.o_spikes = (po > np.random.rand(self.outputs)).astype(int)
        # self.o_spikes = 2*self.o_spikes - 1
        # self.oz = 1 + np.exp(2*zo*self.o_spikes)

        return self.o_spikes

    def update_weights(self, tderror, state, action, mean_reward):
        # h_grad = self.h_spikes
        # h_grad[np.where(self.h_spikes == -1)] = -2*self.hz
        # h_grad[np.where(self.h_spikes == 1)] = 2*(1-self.hz)
        # self.ih_bias += self.alpha*tderror*h_grad
        # for j in range(self.hidden):
        #     if (self.h_spikes[j] == -1):
        #         self.ih_weights[j] -= self.alpha*tderror*np.outer(2 * self.h_spikes[j]/self.hz[j], self.in_spikes)
        #     else:
        #         self.ih_weights[j] += self.alpha*tderror*np.outer(2 * self.h_spikes[j]/self.hz[j], self.in_spikes)
        # self.ih_weights += self.alpha*tderror*np.outer(2*self.h_spikes/self.hz, self.in_spikes)
        # o_grad = self.o_spikes
        # pdb.set_trace()

        # o_grad[np.where(self.o_spikes) == -1] = -2*self.oz
        # o_grad[np.where(self.o_spikes) == 1] = 2*(1-self.oz)

        # self.ho_bias += self.alpha*tderror*o_grad

        # for i in range(self.outputs):
        #     if i == action:
        #         for j in range(self.hidden):
        #             self.ho_weights[i, j] += self.alpha * \
        #                 tderror*(2*self.o_spikes[i]/self.oz[i])*self.h_spikes[j]
        #     if i != action and tderror > 0:
        #         for j in range(self.hidden):
        #             self.ho_weights[i, j] -= self.alpha * \
        #                 tderror*(2*self.o_spikes[i]/self.oz[i])*self.h_spikes[j]
                        
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


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', dest='algorithm', default='sarsa')
    parser.add_argument('--features', dest='features', default='fourier')
    parser.add_argument('--selection', dest='selection', default='egreedy')
    parser.add_argument('--num_trials', dest='num_trials', default=1)
    parser.add_argument('--num_episodes', dest='num_episodes', default=150)
    parser.add_argument('--plot', dest='plot', action='store_true')

    args = parser.parse_args()

    rewards_trials = []

    step_size = 0.00005  # Sarsa, fourier
    epsilon = 0.1

    for i in range(int(args.num_trials)):
        print('Trial:', i)
        td_cp = ActorCritic(order=2, epsilon=epsilon,
                            step_size=step_size, num_states=6)
        rewards = td_cp.run_actor_critic(
            int(args.num_episodes), features='fourier')
        rewards_trials.append(rewards)

    if args.plot:
        episodes = np.linspace(
            0, int(args.num_episodes)-1, int(args.num_episodes))
        rewards_mean = np.mean(rewards_trials, axis=0)
        rewards_std = np.std(rewards_trials, axis=0)
        plt.errorbar(episodes, rewards_mean, rewards_std)
        plt.ylabel('Mean reward')
        plt.xlabel('Number of episodes')
        plt.show()

    f = open('rewards_ac.pkl', 'wb')
    pickle.dump(rewards_trials, f)
    
    h=30
    plt.plot(np.convolve(rewards_mean, np.ones(h)/h, 'valid'))
    plt.title(f'{h}-trial moving average of rewards')
    plt.ylabel('reward')
    plt.xlabel('Number of episodes')
    plt.show()
    
    for i in range(5):
        f = open(f'ih_weights_ac_{i}.pkl', 'wb')
        pickle.dump(td_cp.actors[i].ih_weights, f)
        
    for i in range(5):
        f = open(f'ho_weights_ac_{i}.pkl', 'wb')
        pickle.dump(td_cp.actors[i].ho_weights, f)
    
    # test for 50 episodes
    
    test_rewards = []
    test_episodes = 50
    for i in range(test_episodes):
        state = td_cp.acrobot.reset()
        rt = -1
        count = 0
        if isinstance(state, tuple):
            state = state[0]
        done = False
        while not done:
            # Act using actor

            o_rates = []
            for k in range(len(td_cp.actors)):
                o_spikes = td_cp.actors[k].forward(state, count)
                o_rates.append(o_spikes)
            o_rates = np.array(o_rates)
            action_rates = np.zeros(3)
            for k in range(3):
                action_rates[k] = sum(
                    o_rates[np.where(o_rates[:, k] == 1), k][0])
            action_index = np.argmax(action_rates)
            action = td_cp.e_greedy_action(action_index)

            new_state, reward, terminated, truncated, _ = td_cp.acrobot.step(
                int(action))
            done = terminated or truncated

            if done:
                break

            state = new_state
            count += 1

        print("Reward after %s episodes: %s" % (i, -count))
        test_rewards.append(-count)
            
        h=5
        plt.plot(np.convolve(test_rewards, np.ones(h)/h, 'valid'))
        plt.title(f'{h}-trial moving average of test rewards')
        plt.ylabel('reward')
        plt.xlabel('Number of episodes')
        plt.show()
        
        plt.plot(test_rewards)
        plt.title('Test rewards')
        plt.ylabel('reward')
        plt.xlabel('Number of episodes')
        plt.show()
        
        f = open('test_rewards_ac_acrobot.pkl', 'wb')
        pickle.dump(test_rewards, f)
    