import os
import torch
import torchvision
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd  

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

ROOT_DIR = 'C:/Users/yunes/Мой диск/documents/snn_programs/diehl_acrobot/acrobot'

model = 'diehl_and_cook_2015_5_epochs_1000_hidden_neurons_v3'
data = 'acrobot'

data_dir = 'C:/Users/yunes/Мой диск/documents/snn_programs/diehl_acrobot/acrobot'

# data_path = os.path.join(data_root_dir, 'states_for_scale.json')
params_path = os.path.join(ROOT_DIR, 'params', data, model)
spikes_path = os.path.join(ROOT_DIR, 'spikes', data, model)
curves_path = os.path.join(ROOT_DIR, 'curves', data, model)
results_path = os.path.join(ROOT_DIR, 'results', data, model)
confusion_path = os.path.join(ROOT_DIR, 'confusion', data, model)


class InputLayer:
    def __init__(self, rho=5000):
        self.num_features = 6  
        self.rho = rho
        self.h = np.zeros(self.num_features)
        self.num_points = 100

        self.feature_values = []
        self.h = []

        self.feature_values.append(np.linspace(-1.0, 1.0, self.num_points))
        self.h.append(self.feature_values[0][1] - self.feature_values[0][0])

        self.feature_values.append(np.linspace(-1.0, 1.0, self.num_points))
        self.h.append(self.feature_values[1][1] - self.feature_values[1][0])

        self.feature_values.append(np.linspace(-1.0, 1.0, self.num_points))
        self.h.append(self.feature_values[2][1] - self.feature_values[2][0])

        self.feature_values.append(np.linspace(-1.0, 1.0, self.num_points))
        self.h.append(self.feature_values[3][1] - self.feature_values[3][0])

        self.feature_values.append(np.linspace(-12.567, 12.567, self.num_points))
        self.h.append(self.feature_values[4][1] - self.feature_values[4][0])

        self.feature_values.append(np.linspace(-28.274, 28.274, self.num_points))
        self.h.append(self.feature_values[5][1] - self.feature_values[5][0])

        self.h = np.array(self.h).reshape(self.num_features, 1)

        self.values = np.vstack(self.feature_values)
        # self.threshold = 0.1 * np.ones((self.num_features, 1))

    def step(self, state):
        state = np.array(state).reshape(self.num_features, 1)
        diff = (self.values - state) ** 2
        exp = np.exp(-0.5 * diff / (self.h)**2)
        # prob = softmax(exp, axis=1)

        # spikes = (prob > self.threshold).flatten()

        return self.rho * torch.from_numpy(exp.flatten()).type(torch.FloatTensor)

def print_results(results: Dict[str, list]) -> None:
    """
    Выводит последние, средние и максимальные результаты по каждой схеме классификации.

    :param results: Словарь, где ключи - названия схем классификации, а значения - списки точностей.
    """
    print()
    for s in results:
        if s != 'ngram':
            last = results[s][-1]
            mean = torch.mean(torch.tensor(results[s])).item()
            std = torch.std(torch.tensor(results[s])).item()
            best = torch.max(torch.tensor(results[s])).item()

            print(f'Results for scheme "{s}": {last:.2f} (last), {mean:.2f} (mean), {std:.2f} (std.), {best:.2f} (best)')


def update_curves(curves: Dict[str, list], actions: torch.Tensor,
                  n_classes: int, **kwargs) -> Tuple[Dict[str, list], Dict[str, torch.Tensor]]:
    """
    Обновляет кривые точности для каждой схемы классификации.

    :param curves: Словарь, где ключи - названия схем классификации, а значения - списки точностей.
    :param actions: Одномерный `torch.Tensor` с действиями.
    :param n_classes: Количество классов действий.
    :param kwargs: Дополнительные аргументы для функций оценки классификации.
    :return: Обновлённые кривые точности и предсказания.
    """
    predictions = {}
    for scheme in curves:
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

        predictions[scheme] = prediction
        accuracy = torch.sum(actions.long() == prediction).float() / len(actions)
        curves[scheme].append(100 * accuracy)

    return curves, predictions


def bit_flip(x: torch.Tensor, p: float) -> torch.Tensor:
    """
    Инвертирует биты в бинарном тензоре с вероятностью p.

    :param x: Бинарный тензор произвольной формы.
    :param p: Вероятность инверсии каждого бита.
    """
    x = x.float()
    i = torch.bernoulli(p * torch.ones_like(x)).byte()
    x = x.byte()
    x[i] = ~x[i]
    return x.float()


def main(seed=0, n_neurons=1000, n_train=16000, n_test=4000, inhib=0.1, lr=1e-6, lr_decay=1.0, time=20, dt=0.5,
         theta_plus=0.0, tc_theta_decay=1e7, intensity=0.5, progress_interval=10, update_interval=50, plot=False,
         train=True, gpu=False, n_epochs=5, w_init_hid=0.5, thresh=-56., refrac=5):

    assert n_train % update_interval == 0 and n_test % update_interval == 0, \
                            'Количество примеров должно делиться на update_interval'

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
    n_classes = 3  

    data_path = os.path.join(data_dir, 'acrobot_state_action_dataset.csv')  
    
    for path in [params_path, spikes_path, curves_path, results_path, confusion_path]:
        if not os.path.isdir(path):
            os.makedirs(path)

    # Создание сети.
    if train:
        network = DiehlAndCook2015v2(
            n_inpt=600, 
            n_neurons=n_neurons, inh=inhib, dt=dt, norm=None,
            theta_plus=theta_plus, tc_theta_decay=tc_theta_decay, nu=[0.25 * lr, lr],
            w_init_hid=w_init_hid, thresh=thresh, refrac=refrac,
        )
    else:
        network = load(os.path.join(params_path, model_name + '.pt'))
        network.connections['X', 'Y'].update_rule = NoOp(
            connection=network.connections['X', 'Y'], nu=network.connections['X', 'Y'].nu
        )
        network.layers['Y'].tc_theta_decay = torch.tensor(0.0, dtype=torch.int8)
        network.layers['Y'].theta_plus = torch.tensor(0.0, dtype=torch.int8)

    df = pd.read_csv(data_path)
    n_states = min(len(df), 20000)  

    state_arr = np.array(df['state'].apply(eval).tolist())[:n_states]
    action_arr = np.array(df['action'].tolist())[:n_states]

    data = {}

    frac = 1.0
    sample_size = int(frac * len(state_arr))
    sample_indices = np.random.choice(len(state_arr), size=sample_size, replace=False)

    data['data'] = state_arr[sample_indices]
    data['target'] = action_arr[sample_indices]

    X = data['data']
    X = torch.from_numpy(X).type(torch.FloatTensor)
    y = data['target']
    y = torch.from_numpy(y).type(torch.LongTensor) 

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=n_test, shuffle=True, random_state=seed)

    # place_cells = PlaceCell() 
    input_layer = InputLayer()

    if train:
        states, actions = X_train, y_train
    else:
        states, actions = X_val, y_val

    if train:
        assignments = -torch.ones(n_neurons, dtype=torch.long)
        proportions = torch.zeros(n_neurons, n_classes)
        rates = torch.zeros(n_neurons, n_classes)
        ngram_scores = {}
    else:
        path = os.path.join(params_path, '_'.join(['auxiliary', model_name]) + '.pt')
        assignments, proportions, rates, ngram_scores = torch.load(open(path, 'rb'))

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

    if train:
        print('\n Начало обучения.\n')
    else:
        print('\n Начало тестирования.\n')

    inpt_axes = None
    inpt_ims = None
    spike_ims = None
    spike_axes = None
    weights_im = None
    assigns_im = None
    perf_ax = None

    start = t()
    for epoch in range(n_epochs):
        print(f'\nEpoch {epoch + 1}/{n_epochs}')
        spike_record = torch.zeros(update_interval, time, n_neurons)
        full_spike_record = torch.zeros(n_examples, n_neurons).long()

        for i in range(n_examples):
            if i % progress_interval == 0 and i != 0:
                elapsed = t() - start
                print(f'Прогресс: {i} / {n_examples} ({elapsed:.2f} секунд)')
                start = t()

            if i % update_interval == 0 and i > 0:
                if train:
                    network.connections['X', 'Y'].update_rule.nu[1] *= lr_decay

                if i % len(actions) == 0:
                    current_actions = actions[-update_interval:]
                else:
                    current_actions = actions[i % len(states) - update_interval:i % len(states)]

                curves, preds = update_curves(
                    curves, current_actions, n_classes, spike_record=spike_record, assignments=assignments,
                    proportions=proportions, ngram_scores=ngram_scores, n=2
                )
                if i % progress_interval == 0:
                    print_results(curves)

                for scheme in preds:
                    if scheme != 'ngram':
                        predictions[scheme] = torch.cat([predictions[scheme], preds[scheme]], -1)

                to_write = ['train'] + params if train else ['test'] + test_params
                f = '_'.join([str(x) for x in to_write]) + '.pt'
                torch.save((curves, update_interval, n_examples), open(os.path.join(curves_path, f), 'wb'))

                if train:
                    if any([x[-1] > best_accuracy for x in curves.values()]):
                        print('Новая лучшая точность! Сохранение параметров сети на диск.')

                        network.save(os.path.join(params_path, model_name + '.pt'))
                        path = os.path.join(params_path, '_'.join(['auxiliary', model_name]) + '.pt')
                        torch.save((assignments, proportions, rates, ngram_scores), open(path, 'wb'))
                        best_accuracy = max([x[-1] for x in curves.values()])

                    assignments, proportions, rates = assign_labels(spike_record, current_actions, n_classes, rates)

                    # ngram_scores = update_ngram_scores(spike_record, current_actions, n_classes, 2, ngram_scores)

            state = states[i % len(states)]
            rates_for_states = input_layer.step(state)
            sample = poisson(datum=rates_for_states, time=time, dt=dt)
            inpts = {'X': sample}

            network.run(inputs=inpts, time=time, one_step=True)

            spike_record[i % update_interval] = spikes['Y'].get('s').squeeze(1).t().permute(1, 0)
            full_spike_record[i] = spikes['Y'].get('s').squeeze(1).t().permute(1, 0).sum(0).long()

            if plot:
                _spikes = {layer: spikes[layer].get('s') for layer in spikes}
                input_exc_weights = network.connections[('X', 'Y')].w

                spike_ims, spike_axes = plot_spikes(_spikes, ims=spike_ims, axes=spike_axes)
                perf_ax = plot_performance(curves, ax=perf_ax)

                plt.pause(1e-8)

            network.reset_state_variables() 

        print(f'Прогресс: {n_examples} / {n_examples} (обработано за {(t() - start):.2f} секунд)')

        i += 1

        if i % len(actions) == 0:
            current_actions = actions[-update_interval:]
        else:
            current_actions = actions[i % len(states) - update_interval:i % len(states)]

        curves, preds = update_curves(
            curves, current_actions, n_classes, spike_record=spike_record, assignments=assignments,
            proportions=proportions, ngram_scores=ngram_scores, n=2
        )
        print_results(curves)

        for scheme in preds:
            if scheme != 'ngram':
                predictions[scheme] = torch.cat((predictions[scheme], preds[scheme]), -1)

        if train:
            network.save(os.path.join(params_path, model_name + '_final.pt'))
            path = os.path.join(params_path, '_'.join(['auxiliary', model_name]) + '_final.pt')
            torch.save((assignments, proportions, rates, ngram_scores), open(path, 'wb'))

    if train:
        print('\n Обучение завершено.\n')
    else:
        print('\n Тестирование завершено.\n')

    print('Средние точности:\n')
    for scheme in curves.keys():
        print(f'\t{scheme}: {float(np.mean(curves[scheme])):.2f}')

    to_write = ['train'] + params if train else ['test'] + test_params
    f = '_'.join([str(x) for x in to_write]) + '.pt'
    torch.save((curves, update_interval, n_examples), open(os.path.join(curves_path, f), 'wb'))

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

    confusions = {}
    for scheme in predictions:
        if scheme != 'ngram':
            confusions[scheme] = confusion_matrix(actions, predictions[scheme])

    to_write = ['train'] + params if train else ['test'] + test_params
    f = '_'.join([str(x) for x in to_write]) + '.pt'
    torch.save(confusions, os.path.join(confusion_path, f))

    torch.save(full_spike_record, os.path.join(spikes_path, f))


seed = 0
n_neurons = 1000
n_train = 16000  
n_test = 4000   
inhib = 0.1
lr = 1e-6
lr_decay = 1.0  
time = 20
dt = 0.5
theta_plus = 0.0
tc_theta_decay = 1e7
intensity = 0.5
progress_interval = 10
update_interval = 50    
plot = False
gpu = False
train = True
n_epochs = 15
w_init_hid = 0.5
thresh = -53.
refrac = 0

main(seed=seed, n_neurons=n_neurons, n_train=n_train, n_test=n_test, inhib=inhib, lr=lr, lr_decay=lr_decay, time=time, dt=dt,
     theta_plus=theta_plus, tc_theta_decay=tc_theta_decay, intensity=intensity, progress_interval=progress_interval,
     update_interval=update_interval, plot=plot,
     train=train, gpu=gpu, n_epochs=n_epochs, w_init_hid=w_init_hid, thresh=thresh, refrac=refrac)

print("ok")
