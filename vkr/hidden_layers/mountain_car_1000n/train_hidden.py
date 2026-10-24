import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, REPO_ROOT)  # local modified bindsnet

import torch
import torchvision
import argparse
import json
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import gymnasium as gym

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

# ====== ПУТИ / ИМЕНА ПОД MOUNTAINCAR ======
ROOT_DIR = os.path.join(SCRIPT_DIR, "runs")

model = 'diehl_and_cook_2015_5_epochs_1000_hidden_neurons_v4'
data = 'mountaincar'

data_dir = os.path.join(REPO_ROOT, "data")

params_path = os.path.join(ROOT_DIR, 'params', data, model)
spikes_path = os.path.join(ROOT_DIR, 'spikes', data, model)
curves_path = os.path.join(ROOT_DIR, 'curves', data, model)
results_path = os.path.join(ROOT_DIR, 'results', data, model)
confusion_path = os.path.join(ROOT_DIR, 'confusion', data, model)

# ====== ПАРАМЕТРЫ ENV-EVAL ======
N_EVAL_EPISODES = 20
PROTO_PER_ACTION = 400
MAX_ENV_STEPS = 200

class InputLayer:
    def __init__(self, rho=20000):
        # В MountainCar состояние = [position, velocity]
        self.num_features = 2
        self.rho = rho
        self.num_points = 100

        self.feature_values = []
        self.h = []

        self.feature_values.append(np.linspace(-1.2, 0.6, self.num_points))
        self.h.append(self.feature_values[0][1] - self.feature_values[0][0])

        self.feature_values.append(np.linspace(-0.07, 0.07, self.num_points))
        self.h.append(self.feature_values[1][1] - self.feature_values[1][0])

        self.h = np.array(self.h).reshape(self.num_features, 1)
        self.values = np.vstack(self.feature_values)

    def step(self, state):
        # state: тензор/список длины 2 -> [2,1]
        state = np.array(state).reshape(self.num_features, 1)
        diff = (self.values - state) ** 2
        exp = np.exp(-0.5 * diff / (self.h)**2)
        return self.rho * torch.from_numpy(exp.flatten()).type(torch.FloatTensor)


def print_results(results: Dict[str, list]) -> None:
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
            # заглушка
            # prediction = ngram(kwargs['spike_record'], kwargs['ngram_scores'], n_classes, kwargs['n'])
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
    x = x.float()
    i = torch.bernoulli(p * torch.ones_like(x)).byte()
    x = x.byte()
    x[i] = ~x[i]
    return x.float()


# ============================================================
# ENV-EVAL helpers
# ============================================================
def _cosine_sim(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> float:
    a, b = a.float(), b.float()
    return float(torch.dot(a, b) / (a.norm() * b.norm() + eps))


def _freeze_learning(network) -> Dict:
    saved = {}
    for key, conn in network.connections.items():
        saved[key] = conn.update_rule
        conn.update_rule = NoOp(connection=conn, nu=getattr(conn, "nu", None))
    return saved


def _unfreeze_learning(network, saved: Dict):
    for key, rule in saved.items():
        network.connections[key].update_rule = rule


def _get_eval_monitor(network, time_param, dt_param):
    """Return existing 'Y_spikes_eval' Monitor or attach a new one with full timesteps."""
    timesteps = int(round(time_param / dt_param))
    if 'Y_spikes_eval' not in network.monitors:
        mon = Monitor(network.layers['Y'], state_vars=['s'], time=timesteps)
        network.add_monitor(mon, name='Y_spikes_eval')
    return network.monitors['Y_spikes_eval']


def _build_action_prototypes(network, encoder, states_np, actions_np,
                             time_param, dt_param, n_classes, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    n_neurons = int(network.layers['Y'].n)
    eval_mon = _get_eval_monitor(network, time_param, dt_param)
    protos: Dict[int, torch.Tensor] = {}

    for a in range(n_classes):
        idx = np.where(actions_np == a)[0]
        if len(idx) == 0:
            protos[a] = torch.zeros(n_neurons, dtype=torch.float32)
            continue

        take = min(PROTO_PER_ACTION, len(idx))
        pick = np.random.choice(idx, size=take, replace=False)

        acc = torch.zeros(n_neurons, dtype=torch.float32)
        for j in pick:
            rates = encoder.step(states_np[j])
            spk = poisson(datum=rates, time=time_param, dt=dt_param)

            network.reset_state_variables()
            network.run(inputs={'X': spk}, time=time_param, one_step=False)

            Y_spk = eval_mon.get('s')
            counts = Y_spk.sum(dim=0).squeeze(0).float()
            acc += counts

        proto = acc / float(take)
        norm = proto.norm()
        protos[a] = proto / (norm + 1e-8) if norm > 0 else proto

    return protos


def _pick_action_prototype(counts: torch.Tensor, protos: Dict[int, torch.Tensor]) -> int:
    v = counts.float()
    norm = v.norm()
    if norm > 0:
        v = v / (norm + 1e-8)
    sims = {a: _cosine_sim(v, protos[a]) for a in protos}
    return int(max(sims, key=lambda k: sims[k]))


def eval_in_env(network, encoder, states_np, actions_np,
                time_param, dt_param, n_classes, seed, label='current'):
    t0 = t()
    saved_rules = _freeze_learning(network)
    try:
        protos = _build_action_prototypes(
            network, encoder, states_np, actions_np,
            time_param, dt_param, n_classes, seed
        )
        print(f"    [{label}] prototypes built in {t() - t0:.1f}s")

        eval_mon = _get_eval_monitor(network, time_param, dt_param)
        env = gym.make("MountainCar-v0", render_mode=None)

        ep_returns, ep_lengths, ep_success, final_positions = [], [], [], []

        for ep in range(N_EVAL_EPISODES):
            state, _ = env.reset(seed=seed + ep)
            network.reset_state_variables()

            total_r = 0.0
            success = False
            t_ep = 0

            for t_ep in range(1, MAX_ENV_STEPS + 1):
                rates = encoder.step(state)
                spk = poisson(datum=rates, time=time_param, dt=dt_param)

                network.reset_state_variables()
                network.run(inputs={'X': spk}, time=time_param, one_step=False)

                Y_spk = eval_mon.get('s')
                counts = Y_spk.sum(dim=0).squeeze(0)
                a = _pick_action_prototype(counts, protos)

                next_state, r, terminated, truncated, _ = env.step(int(a))
                total_r += float(r)
                state = next_state

                if terminated:
                    success = True
                    break
                if truncated:
                    break

            ep_returns.append(total_r)
            ep_lengths.append(t_ep)
            ep_success.append(int(success))
            final_positions.append(float(state[0]))

        env.close()
    finally:
        _unfreeze_learning(network, saved_rules)

    metrics = {
        'success_rate': float(np.mean(ep_success)),
        'mean_return': float(np.mean(ep_returns)),
        'std_return': float(np.std(ep_returns)),
        'mean_length': float(np.mean(ep_lengths)),
        'mean_final_pos': float(np.mean(final_positions)),
        'best_final_pos': float(np.max(final_positions)),
        'eval_time_sec': float(t() - t0),
    }
    print(
        f"    [{label}] success={metrics['success_rate']:.2f}  "
        f"mean_ret={metrics['mean_return']:.1f}  "
        f"mean_len={metrics['mean_length']:.1f}  "
        f"mean_pos={metrics['mean_final_pos']:.3f}  "
        f"({metrics['eval_time_sec']:.1f}s)"
    )
    return metrics


def _snapshot_network(network) -> Dict:
    """Save X->Y weights and Y theta of the running network."""
    snap = {
        'w_xy': network.connections[('X', 'Y')].w.detach().clone(),
    }
    if hasattr(network.layers['Y'], 'theta') and torch.is_tensor(network.layers['Y'].theta):
        snap['theta_y'] = network.layers['Y'].theta.detach().clone()
    return snap


def _restore_network(network, snap: Dict):
    network.connections[('X', 'Y')].w.data.copy_(snap['w_xy'])
    if 'theta_y' in snap and hasattr(network.layers['Y'], 'theta'):
        network.layers['Y'].theta.data.copy_(snap['theta_y'])


def _load_into_network(network, ckpt_path: str):
    """Load X->Y weights and Y theta from a v2-style network.save() file into running network."""
    other = load(ckpt_path)
    network.connections[('X', 'Y')].w.data.copy_(
        other.connections[('X', 'Y')].w.detach()
    )
    if hasattr(other.layers['Y'], 'theta') and hasattr(network.layers['Y'], 'theta'):
        network.layers['Y'].theta.data.copy_(other.layers['Y'].theta.detach())


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
    n_classes = 3  # действия MountainCar: 0,1,2

    # ====== ЧТЕНИЕ ДАТАСЕТА MOUNTAINCAR ======
    data_path = os.path.join(data_dir, 'mountaincar_state_action.csv')

    for path in [params_path, spikes_path, curves_path, results_path, confusion_path]:
        if not os.path.isdir(path):
            os.makedirs(path)

    # ====== СЕТЬ: 2 фичи * 100 точек = 200 входных узлов ======
    if train:
        network = DiehlAndCook2015v2(
            n_inpt=200,
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

    # ====== читаем flat и nested датасеты ======
    df = pd.read_csv(data_path)
    n_states = min(len(df), 20000)

    if {'position', 'velocity'}.issubset(df.columns):
        state_arr = df[['position', 'velocity']].values[:n_states]
    else:
        # ожидание колонки 'state' со строковым списком
        state_arr = np.array(df['state'].apply(eval).tolist())[:n_states]
    action_arr = np.array(df['action'].tolist())[:n_states]

    data = {}
    frac = 1.0
    sample_size = int(frac * len(state_arr))
    sample_indices = np.random.choice(len(state_arr), size=sample_size, replace=False)

    data['data'] = state_arr[sample_indices]
    data['target'] = action_arr[sample_indices]

    X = torch.from_numpy(data['data']).type(torch.FloatTensor)          # shape [N, 2]
    y = torch.from_numpy(data['target']).type(torch.LongTensor)         # shape [N]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=n_test, shuffle=True, random_state=seed)

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
    predictions = {scheme: torch.Tensor().long() for scheme in curves.keys()}

    if train:
        best_accuracy = 0

    spikes = {}
    for layer in set(network.layers):
        spikes[layer] = Monitor(network.layers[layer], state_vars=['s'], time=int(time*dt))
        network.add_monitor(spikes[layer], name='%s_spikes' % layer)

    print('\n Начало обучения.\n' if train else '\n Начало тестирования.\n')

    inpt_axes = None
    inpt_ims = None
    spike_ims = None
    spike_axes = None
    weights_im = None
    assigns_im = None
    perf_ax = None

    # ====== ENV-EVAL: лог по эпохам ======
    env_eval_log = []

    start = t()
    for epoch in range(n_epochs):
        print(f'\nEpoch {epoch + 1}/{n_epochs}')
        spike_record = torch.zeros(update_interval, int(time*dt), n_neurons)
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
            rates_for_states = input_layer.step(state)           # -> [200] firing rates
            sample = poisson(datum=rates_for_states, time=time, dt=dt)
            inpts = {'X': sample}

            network.run(inputs=inpts, time=time, one_step=False)

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

        # =====================================================
        # ENV-EVAL: проверяем CURRENT и BEST после каждой эпохи
        # =====================================================
        if train:
            states_np_eval = states.cpu().numpy()
            actions_np_eval = actions.cpu().numpy()

            print(f"\n[ENV-EVAL] Epoch {epoch + 1}: testing CURRENT network ({N_EVAL_EPISODES} eps)...")
            cur_metrics = eval_in_env(
                network=network,
                encoder=input_layer,
                states_np=states_np_eval,
                actions_np=actions_np_eval,
                time_param=time, dt_param=dt, n_classes=n_classes,
                seed=seed, label='current',
            )

            best_path = os.path.join(params_path, model_name + '.pt')
            if os.path.exists(best_path):
                print(f"[ENV-EVAL] Epoch {epoch + 1}: testing BEST checkpoint ({N_EVAL_EPISODES} eps)...")
                snap = _snapshot_network(network)
                _load_into_network(network, best_path)
                best_metrics = eval_in_env(
                    network=network,
                    encoder=input_layer,
                    states_np=states_np_eval,
                    actions_np=actions_np_eval,
                    time_param=time, dt_param=dt, n_classes=n_classes,
                    seed=seed, label='best',
                )
                _restore_network(network, snap)
            else:
                print(f"[ENV-EVAL] Epoch {epoch + 1}: no best checkpoint ({best_path}) - skipping BEST")
                best_metrics = {k: None for k in cur_metrics}

            env_eval_log.append({
                'epoch': epoch + 1,
                'best_accuracy_so_far': float(best_accuracy),
                **{f'cur_{k}': v for k, v in cur_metrics.items()},
                **{f'best_{k}': v for k, v in best_metrics.items()},
            })

            # сохранение лога env-eval (отдельный JSON)
            with open(os.path.join(results_path, 'env_eval_log.json'), 'w', encoding='utf-8') as fh:
                json.dump(env_eval_log, fh, ensure_ascii=False, indent=2)

    print('\n Обучение завершено.\n' if train else '\n Тестирование завершено.\n')

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
                    'mean_ngram,max_all_activity,max_ngram\n'
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
lr = 1e-5
lr_decay = 1.0
time = 40
dt = 0.5
theta_plus = 0.0
tc_theta_decay = 1e7
intensity = 0.5
progress_interval = 10
update_interval = 50
plot = False
gpu = False
train = True
n_epochs = 50
w_init_hid = 0.5
thresh = -56.
refrac = 5

main(seed=seed, n_neurons=n_neurons, n_train=n_train, n_test=n_test, inhib=inhib, lr=lr, lr_decay=lr_decay, time=time, dt=dt,
     theta_plus=theta_plus, tc_theta_decay=tc_theta_decay, intensity=intensity, progress_interval=progress_interval,
     update_interval=update_interval, plot=plot,
     train=train, gpu=gpu, n_epochs=n_epochs, w_init_hid=w_init_hid, thresh=thresh, refrac=refrac)

print("ok")
