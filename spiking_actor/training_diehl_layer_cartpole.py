import os
import torch
import torchvision
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt

from typing import List, Tuple, Dict
from torchvision.datasets import MNIST

from time import time as t
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from bindsnet.network import load
from bindsnet.learning import NoOp
from bindsnet.datasets import MNIST
from bindsnet.encoding import poisson
from bindsnet.network import load
from bindsnet.network.monitors import Monitor
from bindsnet.models import DiehlAndCook2015v2
from bindsnet.evaluation import assign_labels, update_ngram_scores, all_activity, proportion_weighting, ngram, logreg_predict
from bindsnet.utils import get_square_weights, get_square_assignments
from bindsnet.analysis.plotting import plot_input, plot_spikes, plot_weights, plot_assignments, plot_performance

from scipy.special import softmax

ROOT_DIR = 'C:/Users/yunes/Мой диск/documents/snn_programs/diehl_cartpole/cartpole/cartpole_1000_neurons'
data_root_dir = 'C:/Users/yunes/Мой диск/documents/snn_programs/diehl_cartpole/cartpole'


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

def print_results(results: Dict[str, list]) -> None:
    # language=rst
    """
    Prints out latest, mean, and maximum results per classification scheme.

    :param results: Mapping from name of classification scheme to list of accuracy evaluations.
    """
    print()
    for s in results:
        if s != "ngram":
            last = results[s][-1]
            mean = torch.mean(torch.tensor(results[s])).item()
            std = torch.std(torch.tensor(results[s])).item()
            best = torch.max(torch.tensor(results[s])).item()

            print(f'Results for scheme "{s}": {last:.2f} (last), {mean:.2f} (mean), {std:.2f} (std.), {best:.2f} (best)')


def update_curves(curves: Dict[str, list], actions: torch.Tensor,
                  n_classes: int, **kwargs) -> Tuple[Dict[str, list], Dict[str, torch.Tensor]]:
    # language=rst
    """
    Updates accuracy curves for each classification scheme.

    :param curves: Mapping from name of classification scheme to list of accuracy evaluations.
    :param actions: One-dimensional ``torch.Tensor`` of integer data actions.
    :param n_classes: Number of data categories.
    :param kwargs: Additional keyword arguments for classification scheme evaluation functions.
    :return: Updated accuracy curves and predictions.
    """
    predictions = {}
    for scheme in curves:
        # Branch based on name of classification scheme
        if scheme == 'all':
            spike_record = kwargs['spike_record']
            assignments = kwargs['assignments']

            prediction = all_activity(spike_record, assignments, n_classes)
        elif scheme == 'proportion':
            spike_record = kwargs['spike_record']
            assignments = kwargs['assignments']
            proportions = kwargs['proportions']

            prediction = proportion_weighting(spike_record, assignments, proportions, n_classes)
        elif scheme == 'ngram':
            spike_record = kwargs['spike_record']
            ngram_scores = kwargs['ngram_scores']
            n = kwargs['n']

            # prediction = ngram(spike_record, ngram_scores, n_classes, n)
            prediction = 0
        elif scheme == 'logreg':
            full_spike_record = kwargs['full_spike_record']
            logreg = kwargs['logreg']

            prediction = logreg_predict(spikes=full_spike_record, logreg=logreg)

        else:
            raise NotImplementedError

        # Compute accuracy with current classification scheme.
        predictions[scheme] = prediction
        accuracy = torch.sum(actions.long() == prediction).float() / len(actions)
        curves[scheme].append(100 * accuracy)

    return curves, predictions


def bit_flip(x: torch.Tensor, p: float) -> torch.Tensor:
    # language=rst
    """
    Takes a binary tensor and flips each entry with probability p.

    :param x: Arbitrarily shaped binary tensor.
    :param p: Bit flip probability.
    """
    x = x.float()
    i = torch.bernoulli(p * torch.ones_like(x)).byte()
    x = x.byte()
    x[i] = ~x[i]
    return x.float()


model = 'diehl_and_cook_2015_5_epochs_1000_hidden_neurons_400_input_neurons'
data = 'cartpole'

data_path = os.path.join(data_root_dir, 'states_for_scale.json')
params_path = os.path.join(ROOT_DIR, 'params', data, model)
spikes_path = os.path.join(ROOT_DIR, 'spikes', data, model)
curves_path = os.path.join(ROOT_DIR, 'curves', data, model)
results_path = os.path.join(ROOT_DIR, 'results', data, model)
confusion_path = os.path.join(ROOT_DIR, 'confusion', data, model)

for path in [params_path, spikes_path, curves_path, results_path, confusion_path]:
    if not os.path.isdir(path):
        os.makedirs(path)


def main(seed=0, 
         n_neurons=1000,
         n_train=15600,
         n_test=3900, 
         inhib=100, 
         lr=1e-2, 
         lr_decay=1, 
         time=20, 
         dt=0.5,
         theta_plus=0.0, 
         tc_theta_decay=1e7, 
         intensity=1, 
         progress_interval=10, 
         update_interval=50, 
         plot=False,
         train=True, 
         gpu=False, 
         n_epochs=5, 
         w_init_hid=0.5, 
         thresh=-58.
         ):

    assert n_train % update_interval == 0 and n_test % update_interval == 0, \
                            'No. examples must be divisible by update_interval'

    params = [
        seed, n_neurons, n_train, inhib, lr, lr_decay, time, dt, theta_plus,
        tc_theta_decay, intensity, progress_interval, update_interval
    ]

    test_params = [
        seed, n_neurons, n_train, n_test, inhib, lr, lr_decay, time, dt,
        theta_plus, tc_theta_decay, intensity, progress_interval, update_interval
    ]

    model_name = '_'.join([str(x) for x in params])

    np.random.seed(seed)

    if gpu:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        torch.cuda.manual_seed_all(seed)
    else:
        torch.manual_seed(seed)

    n_examples = n_train if train else n_test
    n_sqrt = int(np.ceil(np.sqrt(n_neurons)))
    n_classes = 2

    # Build network.
    if train:
        network = DiehlAndCook2015v2(
            n_inpt=60, n_neurons=n_neurons, inh=inhib, dt=dt, norm=None,
            theta_plus=theta_plus, tc_theta_decay=tc_theta_decay, nu=[0.25 * lr, lr],
            w_init_hid=w_init_hid, thresh=thresh,
            initial_model_path=None,
            one_spike_output=False, refrac=5,
        )

    else:
        network = load(os.path.join(params_path, model_name + '.pt'))
        network.connections['X', 'Y'].update_rule = NoOp(
            connection=network.connections['X', 'Y'], nu=network.connections['X', 'Y'].nu
        )
        network.layers['Y'].tc_theta_decay = torch.tensor(0.0, dtype=torch.int8)
        network.layers['Y'].theta_plus = torch.tensor(0.0, dtype=torch.int8)


    # Load state-action matrix
    state_action_matrix = json.load(open(data_path, "r")) 
    n_states = min(len(state_action_matrix['state']), 19500)

    state_arr = np.array(state_action_matrix['state'])[:n_states]
    action_arr = np.array(state_action_matrix['action'])[:n_states]

    data = {}

    frac = 1.0
    sample_size = int(frac * len(state_arr))
    sample_indices = np.random.choice(len(state_arr), size=sample_size, replace=False)


    data['data'] = state_arr[sample_indices]
    data['target'] = action_arr[sample_indices]

    X = data['data']
    X = torch.from_numpy(X).type(torch.FloatTensor)
    y = data['target']
    y = torch.from_numpy(y).type(torch.FloatTensor)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

    input_layer = InputLayer()

    if train:
        states, actions = X_train, y_train
    else:
        states, actions = X_val, y_val

    # Neuron assignments and spike proportions.
    if train:
        assignments = -torch.ones_like(torch.Tensor(n_neurons))
        proportions = torch.zeros_like(torch.Tensor(n_neurons, n_classes))
        rates = torch.zeros_like(torch.Tensor(n_neurons, n_classes))
        ngram_scores = {}
    else:
        path = os.path.join(params_path, '_'.join(['auxiliary', model_name]) + '.pt')
        assignments, proportions, rates, ngram_scores = torch.load(open(path, 'rb'))

    # Sequence of accuracy estimates.
    curves = {'all': [], 'proportion': [], 'ngram': []}
    predictions = {
        scheme: torch.Tensor().long() for scheme in curves.keys()
    }

    if train:
        best_accuracy = 0

    spikes = {}
    for layer in set(network.layers):
        spikes[layer] = Monitor(network.layers[layer], state_vars=['s'], time=time)
        network.add_monitor(spikes[layer], name='%s_spikes' % layer)

    # Train the network.
    if train:
        print('\nBegin training.\n')
    else:
        print('\nBegin test.\n')

    inpt_axes = None
    inpt_ims = None
    spike_ims = None
    spike_axes = None
    weights_im = None
    assigns_im = None
    perf_ax = None

    start = t()
    for epoch in range(n_epochs):
        # Record spikes during the simulation.
        spike_record = torch.zeros(update_interval, time, n_neurons)
        full_spike_record = torch.zeros(n_examples, n_neurons).long()

        for i in range(n_examples):
            if i % progress_interval == 0:
                print(f'Progress: {i} / {n_examples} ({t() - start:.4f} seconds)')
                start = t()

            if i % update_interval == 0 and i > 0:
                if train:
                    network.connections['X', 'Y'].update_rule.nu[1] *= lr_decay

                if i % len(actions) == 0:
                    current_actions = actions[-update_interval:]
                else:
                    current_actions = actions[i % len(states) - update_interval:i % len(states)]

                # Update and print accuracy evaluations.
                curves, preds = update_curves(
                    curves, current_actions, n_classes, spike_record=spike_record, assignments=assignments,
                    proportions=proportions, ngram_scores=ngram_scores, n=2
                )
                if i % progress_interval == 0:
                    print_results(curves)

                for scheme in preds:
                    if scheme != "ngram":
                        predictions[scheme] = torch.cat([predictions[scheme], preds[scheme]], -1)

                # Save accuracy curves to disk.
                to_write = ['train'] + params if train else ['test'] + params
                f = '_'.join([str(x) for x in to_write]) + '.pt'
                torch.save((curves, update_interval, n_examples), open(os.path.join(curves_path, f), 'wb'))

                if train:
                    if any([x[-1] > best_accuracy for x in curves.values()]):
                        print('New best accuracy! Saving network parameters to disk.')

                        # Save network to disk.
                        network.save(os.path.join(params_path, model_name + '.pt'))
                        path = os.path.join(params_path, '_'.join(['auxiliary', model_name]) + '.pt')
                        torch.save((assignments, proportions, rates, ngram_scores), open(path, 'wb'))
                        best_accuracy = max([x[-1] for x in curves.values()])

                    # Assign labels to excitatory layer neurons.
                    assignments, proportions, rates = assign_labels(spike_record, current_actions, n_classes, rates)

                    # Compute ngram scores.
                    ngram_scores = {}
                    # ngram_scores = update_ngram_scores(spike_record, current_actions, n_classes, 2, ngram_scores)

                # print()

            # Get next input sample.
            state = states[i % len(states)]
            rates_for_states = input_layer.step(state)
            sample = poisson(datum=rates_for_states, time=time, dt=dt)
            inpts = {'X': sample}

            # Run the network on the input.
            network.run(inputs=inpts, time=time)

            # retries = 0
            # while spikes['Y'].get('s').sum() < 1 and retries < 3:
            #     retries += 1
            #     rates_for_states *= 2
            #     sample = poisson(datum=rates_for_states, time=time, dt=dt)
            #     inpts = {'X': sample}
            #     network.run(inputs=inpts, time=time)

            # Add to spikes recording.
            spike_record[i % update_interval] = spikes['Y'].get('s').squeeze(1).t().permute(1, 0)
            full_spike_record[i] = spikes['Y'].get('s').squeeze(1).t().permute(1, 0).sum(0).long()

            # Optionally plot various simulation information.
            if plot:
                # _input = image.view(28, 28)
                # reconstruction = inpts['X'].view(time, 784).sum(0).view(28, 28)
                _spikes = {layer: spikes[layer].get('s') for layer in spikes}
                input_exc_weights = network.connections[('X', 'Y')].w
                # square_weights = get_square_weights(input_exc_weights.view(1715, n_neurons), n_sqrt, )
                # square_assignments = get_square_assignments(assignments, n_sqrt)

                # inpt_axes, inpt_ims = plot_input(_input, reconstruction, label=labels[i], axes=inpt_axes, ims=inpt_ims)
                spike_ims, spike_axes = plot_spikes(_spikes, ims=spike_ims, axes=spike_axes)
                # weights_im = plot_weights(square_weights, im=weights_im)
                # assigns_im = plot_assignments(square_assignments, im=assigns_im)
                perf_ax = plot_performance(curves, ax=perf_ax)

                plt.pause(1e-8)

            network.reset_state_variables()  # Reset state variables.

    print(f'Progress: {n_examples} / {n_examples} ({t() - start:.4f} seconds)')

    i += 1

    if i % len(actions) == 0:
        current_actions = actions[-update_interval:]
    else:
        current_actions = actions[i % len(states) - update_interval:i % len(states)]

    # Update and print accuracy evaluations.
    curves, preds = update_curves(
        curves, current_actions, n_classes, spike_record=spike_record, assignments=assignments,
        proportions=proportions, ngram_scores=ngram_scores, n=2
    )
    print_results(curves)

    for scheme in preds:
        predictions[scheme] = torch.cat((predictions[scheme], preds[scheme]), -1)

    if train:
        # if any([x[-1] > best_accuracy for x in curves.values()]):
            # print('New best accuracy! Saving network parameters to disk.')

        # Save network to disk.
        network.save(os.path.join(params_path, model_name + '_final.pt'))
        path = os.path.join(params_path, '_'.join(['auxiliary', model_name]) + '_final.pt')
        torch.save((assignments, proportions, rates, ngram_scores), open(path, 'wb'))

    if train:
        print('\nTraining complete.\n')
    else:
        print('\nTest complete.\n')

    print('Average accuracies:\n')
    for scheme in curves.keys():
        print('\t%s: %.2f' % (scheme, float(np.mean(curves[scheme]))))

    # Save accuracy curves to disk.
    to_write = ['train'] + params if train else ['test'] + params
    f = '_'.join([str(x) for x in to_write]) + '.pt'
    torch.save((curves, update_interval, n_examples), open(os.path.join(curves_path, f), 'wb'))

    # Save results to disk.
    results = [
        np.mean(curves['all']), np.mean(curves['proportion']), np.mean(curves['ngram']),
        np.max(curves['all']), np.max(curves['proportion']), np.max(curves['ngram'])
    ]

    to_write = params + results if train else test_params + results
    to_write = [str(x) for x in to_write]
    name = 'train.csv' if train else 'test.csv'

    if not os.path.isfile(os.path.join(results_path, name)):
        with open(os.path.join(results_path, name), 'w') as f:
            if train:
                f.write(
                    'random_seed,n_neurons,n_train,inhib,lr,lr_decay,time,timestep,theta_plus,tc_theta_decay,intensity,'
                    'progress_interval,update_interval,mean_all_activity,mean_proportion_weighting,'
                    'mean_ngram,max_all_activity,max_proportion_weighting,max_ngram\n'
                )
            else:
                f.write(
                    'random_seed,n_neurons,n_train,n_test,inhib,lr,lr_decay,time,timestep,theta_plus,tc_theta_decay,'
                    'intensity,progress_interval,update_interval,mean_all_activity,mean_proportion_weighting,'
                    'mean_ngram,max_all_activity,max_proportion_weighting,max_ngram\n'
                )

    with open(os.path.join(results_path, name), 'a') as f:
        f.write(','.join(to_write) + '\n')

    if actions.numel() > n_examples:
        actions = actions[:n_examples]
    else:
        while actions.numel() < n_examples:
            if 2 * actions.numel() > n_examples:
                actions = torch.cat([actions, actions[:n_examples - actions.numel()]])
            else:
                actions = torch.cat([actions, actions])

    # Compute confusion matrices and save them to disk.
    confusions = {}
    for scheme in predictions:
        confusions[scheme] = confusion_matrix(actions, predictions[scheme])

    to_write = ['train'] + params if train else ['test'] + test_params
    f = '_'.join([str(x) for x in to_write]) + '.pt'
    torch.save(confusions, os.path.join(confusion_path, f))

    # Save full spike record to disk.
    torch.save(full_spike_record, os.path.join(spikes_path, f))


seed = 0
n_neurons = 1000
n_train = 15600
n_test = 3900
inhib = 0.1
lr = 1e-5
lr_decay = 1.0 # previous: 0.99
time = 20
dt = 0.5
theta_plus = 0.0
tc_theta_decay = 1e7
intensity = 0.5
progress_interval = 10 # previous: 10
update_interval = 50 # previous: 50
plot=False
gpu=False
train=True
n_epochs=5
w_init_hid = 0.5
thresh=-58.
    

main(seed=seed, n_neurons=n_neurons, n_train=n_train, n_test=n_test, inhib=inhib, lr=lr, lr_decay=lr_decay, time=time, dt=dt,
         theta_plus=theta_plus, tc_theta_decay=tc_theta_decay, intensity=intensity, progress_interval=progress_interval,
         update_interval=update_interval, plot=plot,
         train=train, gpu=gpu, n_epochs=n_epochs, w_init_hid=w_init_hid, thresh=thresh)

print("ok")
