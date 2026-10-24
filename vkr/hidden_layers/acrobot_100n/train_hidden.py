"""
STDP pretraining of the actor hidden layer (DiehlAndCook2015v2) for Acrobot, 100 neurons.

  - The model is checkpointed after every epoch (and at the end)
  - Per-epoch dataset accuracy + env eval are recorded to JSON/CSV

Output dir:
  RESULTS_DIR/<RUN_NAME>/
    epoch_<k>.pt          # bindsnet network state after epoch k
    final.pt              # = last epoch
    metrics.json          # per-epoch metrics (cumulative, rewritten each epoch)
    metrics.csv           # same, flat csv
"""

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, REPO_ROOT)  # local modified bindsnet

import ast
import csv
import json
import random
import warnings
from time import time as now
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import gymnasium as gym
from sklearn.model_selection import train_test_split

from bindsnet.encoding import poisson
from bindsnet.network.monitors import Monitor
from bindsnet.models import DiehlAndCook2015v2
from bindsnet.evaluation import assign_labels, all_activity, proportion_weighting

warnings.filterwarnings("ignore")

# =========================================================
# CONFIG
# =========================================================
DATA_PATH = os.path.join(REPO_ROOT, "data", "acrobot_state_action.csv")

N_NEURONS = 100
N_EPOCHS = 100
RUN_NAME = f"fixed_config_{N_NEURONS}neurons_{N_EPOCHS}epochs"
RESULTS_DIR = os.path.join(SCRIPT_DIR, "runs", RUN_NAME)

N_EVAL_EPISODES = 20
N_PROTO_PER_ACTION = 100   # per action class

# Training hyperparams
SEED = 0
N_TRAIN = 16000
N_TEST = 4000
INHIB = 0.1
LR = 1e-7
LR_DECAY = 1.0
TIME = 20          # ms
DT = 0.5
THETA_PLUS = 0.0
TC_THETA_DECAY = 1e7
UPDATE_INTERVAL = 50
PROGRESS_INTERVAL = 500
W_INIT_HID = 1.0
THRESH = -53.0
REFRAC = 0
CENTERS_PER_FEATURE = (12, 12, 12, 12, 10, 10)
SIGMA_SCALE = 0.6
RHO = 500.0
NORMALIZE_PER_FEATURE = True
N_CLASSES = 3
Q_LOW = 0.01
Q_HIGH = 0.99
MAX_STEPS_ENV = 500

TIMESTEPS = int(round(TIME / DT))   # = 40


# =========================================================
# ENCODER
# =========================================================
class GRFInputLayer:
    def __init__(
        self,
        centers_per_feature=(24, 24, 24, 24, 20, 20),
        feature_ranges=None,
        sigma_scale=0.6,
        rho=500.0,
        normalize_per_feature=True,
        clip_rates: Optional[Tuple[float, float]] = None,
        device="cpu",
    ):
        self.num_features = 6
        self.centers_per_feature = tuple(centers_per_feature)
        self.sigma_scale = float(sigma_scale)
        self.rho = float(rho)
        self.normalize_per_feature = bool(normalize_per_feature)
        self.clip_rates = clip_rates
        self.device = device

        if feature_ranges is None:
            feature_ranges = [
                (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0),
                (-12.567, 12.567), (-28.274, 28.274),
            ]
        self.feature_ranges = feature_ranges

        self.centers: List[np.ndarray] = []
        self.sigmas: List[np.ndarray] = []
        start = 0
        for i in range(self.num_features):
            low, high = self.feature_ranges[i]
            n_c = self.centers_per_feature[i]
            c = np.linspace(low, high, n_c, dtype=np.float32)
            spacing = (c[1] - c[0]) if n_c > 1 else max(high - low, 1.0)
            sigma = max(self.sigma_scale * spacing, 1e-6)
            self.centers.append(c)
            self.sigmas.append(np.full_like(c, sigma))
            start += n_c
        self.n_inpt = start

    def encode_numpy(self, state: np.ndarray) -> np.ndarray:
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        outputs = []
        for i in range(self.num_features):
            resp = np.exp(-0.5 * ((state[i] - self.centers[i]) / self.sigmas[i]) ** 2)
            if self.normalize_per_feature:
                resp = resp / (resp.sum() + 1e-8)
            outputs.append(resp)
        rates = np.concatenate(outputs) * self.rho
        if self.clip_rates is not None:
            rates = np.clip(rates, self.clip_rates[0], self.clip_rates[1])
        return rates.astype(np.float32)

    def step(self, state) -> torch.Tensor:
        rates = self.encode_numpy(np.asarray(state, dtype=np.float32))
        return torch.tensor(rates, dtype=torch.float32, device=self.device)


# =========================================================
# UTILS
# =========================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_state_column(x):
    if isinstance(x, str):
        return ast.literal_eval(x)
    return x


def build_feature_ranges_from_data(X_np: np.ndarray, q_low=0.01, q_high=0.99):
    ranges = [(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)]
    for j in [4, 5]:
        low = float(np.quantile(X_np[:, j], q_low))
        high = float(np.quantile(X_np[:, j], q_high))
        if abs(high - low) < 1e-8:
            high = low + 1e-3
        margin = 0.05 * (high - low)
        ranges.append((low - margin, high + margin))
    return ranges


def get_spike_counts(network: DiehlAndCook2015v2) -> torch.Tensor:
    Y_spk = network.monitors["Y_spikes"].get("s")
    return Y_spk.sum(dim=0)[0].float()


# =========================================================
# ENV EVALUATION (prototype policy)
# =========================================================
def evaluate_in_env(
    network: DiehlAndCook2015v2,
    input_layer: GRFInputLayer,
    states_np: np.ndarray,
    actions_np: np.ndarray,
    n_eval_episodes: int,
    seed: int,
    device: str,
) -> Dict:
    set_seed(seed)

    protos = {}
    for a in range(N_CLASSES):
        idx = np.where(actions_np == a)[0]
        take = min(N_PROTO_PER_ACTION, len(idx))
        pick = np.random.choice(idx, size=take, replace=False)

        acc = None
        for j in pick:
            rates = input_layer.step(states_np[j]).to(device)
            spk_X = poisson(datum=rates, time=TIMESTEPS, dt=DT)
            network.reset_state_variables()
            network.run(inputs={"X": spk_X}, time=TIME, one_step=False, learning=False)
            counts = get_spike_counts(network).detach()
            acc = counts if acc is None else (acc + counts)

        proto = acc / float(take)
        norm = proto.norm()
        protos[a] = proto / (norm + 1e-8) if norm > 0 else proto

    env = gym.make("Acrobot-v1", render_mode=None)
    ep_returns, ep_lengths, ep_success = [], [], []

    for ep in range(n_eval_episodes):
        state, _ = env.reset(seed=seed + ep)
        total_r = 0.0
        success = False

        for t in range(1, MAX_STEPS_ENV + 1):
            rates = input_layer.step(np.array(state, dtype=np.float32)).to(device)
            spk_X = poisson(datum=rates, time=TIMESTEPS, dt=DT)
            network.reset_state_variables()
            network.run(inputs={"X": spk_X}, time=TIME, one_step=False, learning=False)
            counts = get_spike_counts(network).detach()

            v = counts / (counts.norm() + 1e-8) if counts.norm() > 0 else counts
            sims = {aa: float(torch.dot(v, protos[aa])) for aa in protos}
            a_star = max(sims, key=lambda k: sims[k])

            next_state, r, terminated, truncated, _ = env.step(int(a_star))
            total_r += float(r)
            state = next_state

            if terminated:
                success = True
                break
            if truncated:
                break

        ep_returns.append(total_r)
        ep_lengths.append(t)
        ep_success.append(int(success))

    env.close()

    return {
        "mean_return": float(np.mean(ep_returns)),
        "std_return": float(np.std(ep_returns)),
        "success_rate": float(np.mean(ep_success)),
        "mean_length": float(np.mean(ep_lengths)),
        "std_length": float(np.std(ep_lengths)),
        "episode_returns": ep_returns,
        "episode_lengths": ep_lengths,
    }


# =========================================================
# TRAIN
# =========================================================
def write_metrics(results_dir: str, n_neurons: int, epoch_results: List[Dict]):
    """(Re)write metrics.json and metrics.csv after each epoch."""
    json_path = os.path.join(results_dir, "metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"n_neurons": n_neurons, "epochs": epoch_results}, f, ensure_ascii=False, indent=2)

    csv_path = os.path.join(results_dir, "metrics.csv")
    fieldnames = [
        "epoch",
        "mean_acc_all_pct", "mean_acc_proportion_pct",
        "success_rate", "mean_return", "std_return", "mean_length", "std_length",
        "epoch_time_sec",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for ep in epoch_results:
            writer.writerow({k: ep.get(k) for k in fieldnames})


def train(n_neurons: int, n_epochs: int):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    set_seed(SEED)
    device = "cpu"

    df = pd.read_csv(DATA_PATH)
    state_arr = np.array(df["state"].apply(parse_state_column).tolist(), dtype=np.float32)
    action_arr = np.array(df["action"].tolist(), dtype=np.int64)

    X = torch.tensor(state_arr, dtype=torch.float32)
    y = torch.tensor(action_arr, dtype=torch.long)

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=N_TEST, shuffle=True, random_state=SEED, stratify=y
    )
    X_train = X_train[:N_TRAIN]
    y_train = y_train[:N_TRAIN]

    feature_ranges = build_feature_ranges_from_data(X_train.numpy(), q_low=Q_LOW, q_high=Q_HIGH)
    input_layer = GRFInputLayer(
        centers_per_feature=CENTERS_PER_FEATURE,
        feature_ranges=feature_ranges,
        sigma_scale=SIGMA_SCALE,
        rho=RHO,
        normalize_per_feature=NORMALIZE_PER_FEATURE,
        device=device,
    )
    print(f"Encoder n_inpt = {input_layer.n_inpt}")
    print(f"Run dir: {RESULTS_DIR}")
    print(f"n_neurons={n_neurons}, n_epochs={n_epochs}, n_train={N_TRAIN}")

    network = DiehlAndCook2015v2(
        n_inpt=input_layer.n_inpt,
        n_neurons=n_neurons,
        inh=INHIB,
        dt=DT,
        norm=None,
        theta_plus=THETA_PLUS,
        tc_theta_decay=TC_THETA_DECAY,
        nu=[0.25 * LR, LR],
        w_init_hid=W_INIT_HID,
        thresh=THRESH,
        refrac=REFRAC,
        wmin=0.0,
        wmax=3.0,
    )

    for layer in set(network.layers):
        mon = Monitor(network.layers[layer], state_vars=["s"], time=TIMESTEPS)
        network.add_monitor(mon, name=f"{layer}_spikes")

    assignments = -torch.ones(n_neurons, dtype=torch.long)
    proportions = torch.zeros(n_neurons, N_CLASSES)
    rates = torch.zeros(n_neurons, N_CLASSES)

    epoch_results: List[Dict] = []
    n_examples = N_TRAIN
    t_total = now()

    for epoch in range(n_epochs):
        t_epoch_start = now()
        print(f"\n=== Epoch {epoch + 1}/{n_epochs} ===")

        spike_record = torch.zeros(UPDATE_INTERVAL, TIMESTEPS, n_neurons)
        full_spike_record = torch.zeros(n_examples, n_neurons).long()
        all_accs_all: List[float] = []
        all_accs_prop: List[float] = []

        lr_current = LR * (LR_DECAY ** epoch)
        network.connections["X", "Y"].update_rule.nu[1] = lr_current

        for i in range(n_examples):
            if i % PROGRESS_INTERVAL == 0 and i > 0:
                elapsed = now() - t_epoch_start
                print(f"  [{i}/{n_examples}] {elapsed:.1f}s elapsed")

            if i % UPDATE_INTERVAL == 0 and i > 0:
                if i % len(y_train) == 0:
                    cur_actions = y_train[-UPDATE_INTERVAL:]
                else:
                    cur_actions = y_train[i % len(X_train) - UPDATE_INTERVAL: i % len(X_train)]

                acc_all = all_activity(spike_record, assignments, N_CLASSES)
                acc_prop = proportion_weighting(spike_record, assignments, proportions, N_CLASSES)

                acc_all_val = float(torch.sum(cur_actions.long() == acc_all)) / UPDATE_INTERVAL
                acc_prop_val = float(torch.sum(cur_actions.long() == acc_prop)) / UPDATE_INTERVAL
                all_accs_all.append(acc_all_val)
                all_accs_prop.append(acc_prop_val)

                assignments, proportions, rates = assign_labels(
                    spike_record, cur_actions, N_CLASSES, rates
                )

            state = X_train[i % len(X_train)]
            enc_rates = input_layer.step(state.cpu().numpy())
            spk_X = poisson(datum=enc_rates, time=TIMESTEPS, dt=DT)

            network.run(inputs={"X": spk_X}, time=TIME, one_step=False)

            y_spikes = network.monitors["Y_spikes"].get("s").squeeze(1)
            spike_record[i % UPDATE_INTERVAL] = y_spikes
            full_spike_record[i] = y_spikes.sum(0).long()

            network.reset_state_variables()

        # End-of-epoch update
        if n_examples % len(y_train) == 0:
            cur_actions = y_train[-UPDATE_INTERVAL:]
        else:
            cur_actions = y_train[n_examples % len(X_train) - UPDATE_INTERVAL: n_examples % len(X_train)]

        acc_all = all_activity(spike_record, assignments, N_CLASSES)
        acc_prop = proportion_weighting(spike_record, assignments, proportions, N_CLASSES)
        acc_all_val = float(torch.sum(cur_actions.long() == acc_all)) / UPDATE_INTERVAL
        acc_prop_val = float(torch.sum(cur_actions.long() == acc_prop)) / UPDATE_INTERVAL
        all_accs_all.append(acc_all_val)
        all_accs_prop.append(acc_prop_val)

        assignments, proportions, rates = assign_labels(
            spike_record, cur_actions, N_CLASSES, rates
        )

        mean_acc_all = float(np.mean(all_accs_all)) * 100
        mean_acc_prop = float(np.mean(all_accs_prop)) * 100
        t_train = now() - t_epoch_start

        print(f"  Dataset acc — all_activity: {mean_acc_all:.1f}%  proportion: {mean_acc_prop:.1f}%  ({t_train:.1f}s)")

        # Env evaluation
        print(f"  Running env eval ({N_EVAL_EPISODES} episodes) ...")
        t_env_start = now()
        env_metrics = evaluate_in_env(
            network=network,
            input_layer=input_layer,
            states_np=state_arr,
            actions_np=action_arr,
            n_eval_episodes=N_EVAL_EPISODES,
            seed=SEED,
            device=device,
        )
        t_env = now() - t_env_start
        print(
            f"  Env — success: {env_metrics['success_rate']:.2f}  "
            f"mean_return: {env_metrics['mean_return']:.1f}  "
            f"mean_len: {env_metrics['mean_length']:.1f}  ({t_env:.1f}s)"
        )

        # Save checkpoint
        ckpt_path = os.path.join(RESULTS_DIR, f"epoch_{epoch + 1}.pt")
        network.save(ckpt_path)
        print(f"  -> saved {ckpt_path}")

        epoch_results.append({
            "epoch": epoch + 1,
            "mean_acc_all_pct": mean_acc_all,
            "mean_acc_proportion_pct": mean_acc_prop,
            "epoch_time_sec": round(t_train + t_env, 1),
            **env_metrics,
        })
        write_metrics(RESULTS_DIR, n_neurons, epoch_results)

    # Final = copy of last epoch
    network.save(os.path.join(RESULTS_DIR, "final.pt"))

    print(f"\nTotal time: {(now() - t_total)/60:.1f} min")
    print(f"\n{'epoch':>6} {'acc_all%':>10} {'acc_prop%':>10} {'success':>8} {'ret':>8} {'len':>8}")
    print("-" * 56)
    for ep in epoch_results:
        print(
            f"{ep['epoch']:>6} "
            f"{ep['mean_acc_all_pct']:>10.1f} {ep['mean_acc_proportion_pct']:>10.1f} "
            f"{ep['success_rate']:>8.2f} {ep['mean_return']:>8.1f} {ep['mean_length']:>8.1f}"
        )

    return epoch_results


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    train(n_neurons=N_NEURONS, n_epochs=N_EPOCHS)
