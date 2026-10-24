import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, REPO_ROOT)  # local modified bindsnet

import ast
import json
import random
import warnings
from time import time as now
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import pairwise_distances
from scipy.stats import spearmanr

from bindsnet.network import load
from bindsnet.learning import NoOp
from bindsnet.encoding import poisson
from bindsnet.network.monitors import Monitor
from bindsnet.models import DiehlAndCook2015v2
from bindsnet.evaluation import assign_labels, all_activity, proportion_weighting

warnings.filterwarnings("ignore", category=UserWarning)


# =========================================================
# CONFIG
# =========================================================
ROOT_DIR = os.path.join(SCRIPT_DIR, "runs")
DATA_PATH = os.path.join(REPO_ROOT, "data", "acrobot_state_action.csv")
MODEL_NAME = "acrobot_grf_compact_sigma_scale_0_6_w_init_hid_1_5_20neurons_hid_68neurons_input_0420"
DATA_NAME = "acrobot"

PARAMS_PATH = os.path.join(ROOT_DIR, "params", DATA_NAME, MODEL_NAME)
SPIKES_PATH = os.path.join(ROOT_DIR, "spikes", DATA_NAME, MODEL_NAME)
CURVES_PATH = os.path.join(ROOT_DIR, "curves", DATA_NAME, MODEL_NAME)
RESULTS_PATH = os.path.join(ROOT_DIR, "results", DATA_NAME, MODEL_NAME)
CONFUSION_PATH = os.path.join(ROOT_DIR, "confusion", DATA_NAME, MODEL_NAME)
DIAG_PATH = os.path.join(ROOT_DIR, "diagnostics", DATA_NAME, MODEL_NAME)


# =========================================================
# UTILS
# =========================================================
def ensure_dirs(*paths):
    for path in paths:
        os.makedirs(path, exist_ok=True)


def set_seed(seed: int, gpu: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if gpu and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_state_column(x):
    if isinstance(x, str):
        return ast.literal_eval(x)
    return x


# =========================================================
# ENCODERS
# =========================================================
class GRFInputLayer:
    """
    Compact Gaussian Receptive Field encoder.

    Main ideas:
    1) Much fewer centers per feature.
    2) sigma = sigma_scale * spacing.
    3) Velocity ranges are inferred from the dataset.
    4) Per-feature normalization stabilizes total rate.
    """

    def __init__(
        self,
        centers_per_feature=(12, 12, 12, 12, 10, 10),
        feature_ranges=None,
        sigma_scale=0.6,
        rho=1500.0,
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

        assert len(self.centers_per_feature) == self.num_features

        if feature_ranges is None:
            feature_ranges = [
                (-1.0, 1.0),
                (-1.0, 1.0),
                (-1.0, 1.0),
                (-1.0, 1.0),
                (-12.567, 12.567),
                (-28.274, 28.274),
            ]

        assert len(feature_ranges) == self.num_features
        self.feature_ranges = feature_ranges

        self.centers: List[np.ndarray] = []
        self.sigmas: List[np.ndarray] = []
        self.slices: List[Tuple[int, int]] = []

        start = 0
        for i in range(self.num_features):
            low, high = self.feature_ranges[i]
            n_centers = self.centers_per_feature[i]

            c = np.linspace(low, high, n_centers, dtype=np.float32)
            spacing = (c[1] - c[0]) if n_centers > 1 else max(high - low, 1.0)
            sigma = max(self.sigma_scale * spacing, 1e-6)
            s = np.full_like(c, sigma, dtype=np.float32)

            self.centers.append(c)
            self.sigmas.append(s)
            self.slices.append((start, start + n_centers))
            start += n_centers

        self.n_inpt = start

    def encode_numpy(self, state: np.ndarray) -> np.ndarray:
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        assert state.shape[0] == self.num_features

        outputs = []
        for i in range(self.num_features):
            x = state[i]
            c = self.centers[i]
            s = self.sigmas[i]
            resp = np.exp(-0.5 * ((x - c) / s) ** 2)
            if self.normalize_per_feature:
                resp = resp / (resp.sum() + 1e-8)
            outputs.append(resp)

        rates = np.concatenate(outputs, axis=0) * self.rho
        if self.clip_rates is not None:
            rates = np.clip(rates, self.clip_rates[0], self.clip_rates[1])
        return rates.astype(np.float32)

    def step(self, state) -> torch.Tensor:
        rates = self.encode_numpy(np.asarray(state, dtype=np.float32))
        return torch.tensor(rates, dtype=torch.float32, device=self.device)


class OldGRFInputLayer:
    """
    Старый способ кодирования:
    - 100 центров на каждую из 6 фич
    - sigma = spacing
    - фиксированные диапазоны
    - без нормализации по фичам
    """

    def __init__(self, rho=5000.0, num_points=100, device="cpu"):
        self.num_features = 6
        self.rho = rho
        self.num_points = num_points
        self.device = device

        self.feature_values = []
        self.h = []

        self.feature_values.append(np.linspace(-1.0, 1.0, self.num_points, dtype=np.float32))
        self.h.append(self.feature_values[0][1] - self.feature_values[0][0])
        self.feature_values.append(np.linspace(-1.0, 1.0, self.num_points, dtype=np.float32))
        self.h.append(self.feature_values[1][1] - self.feature_values[1][0])
        self.feature_values.append(np.linspace(-1.0, 1.0, self.num_points, dtype=np.float32))
        self.h.append(self.feature_values[2][1] - self.feature_values[2][0])
        self.feature_values.append(np.linspace(-1.0, 1.0, self.num_points, dtype=np.float32))
        self.h.append(self.feature_values[3][1] - self.feature_values[3][0])
        self.feature_values.append(np.linspace(-12.567, 12.567, self.num_points, dtype=np.float32))
        self.h.append(self.feature_values[4][1] - self.feature_values[4][0])
        self.feature_values.append(np.linspace(-28.274, 28.274, self.num_points, dtype=np.float32))
        self.h.append(self.feature_values[5][1] - self.feature_values[5][0])

        self.h = np.array(self.h, dtype=np.float32).reshape(self.num_features, 1)
        self.values = np.vstack(self.feature_values).astype(np.float32)

        self.centers = [v.copy() for v in self.feature_values]
        self.sigmas = [np.full_like(v, fill_value=(v[1] - v[0]), dtype=np.float32) for v in self.feature_values]
        self.feature_ranges = [
            (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0),
            (-12.567, 12.567), (-28.274, 28.274),
        ]
        self.centers_per_feature = (100, 100, 100, 100, 100, 100)
        self.sigma_scale = 1.0
        self.normalize_per_feature = False
        self.n_inpt = self.num_features * self.num_points

    def encode_numpy(self, state):
        state = np.asarray(state, dtype=np.float32).reshape(self.num_features, 1)
        diff = (self.values - state) ** 2
        exp = np.exp(-0.5 * diff / (self.h ** 2))
        rates = self.rho * exp.flatten()
        return rates.astype(np.float32)

    def step(self, state):
        rates = self.encode_numpy(state)
        return torch.tensor(rates, dtype=torch.float32, device=self.device)


# =========================================================
# INPUT ENCODER DIAGNOSTICS
# =========================================================
def build_feature_ranges_from_data(X_train, q_low=0.01, q_high=0.99):
    if isinstance(X_train, torch.Tensor):
        X_np = X_train.cpu().numpy()
    else:
        X_np = np.asarray(X_train)

    ranges = [(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)]
    for j in [4, 5]:
        low = float(np.quantile(X_np[:, j], q_low))
        high = float(np.quantile(X_np[:, j], q_high))
        if abs(high - low) < 1e-8:
            high = low + 1e-3
        margin = 0.05 * (high - low)
        ranges.append((low - margin, high + margin))
    return ranges


def plot_grf_receptive_fields(input_layer, save_path: str):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.ravel()

    for i in range(input_layer.num_features):
        ax = axes[i]
        centers = input_layer.centers[i]
        sigmas = input_layer.sigmas[i]
        low, high = input_layer.feature_ranges[i]
        xs = np.linspace(low, high, 400)

        for c, s in zip(centers, sigmas):
            ys = np.exp(-0.5 * ((xs - c) / s) ** 2)
            ax.plot(xs, ys, alpha=0.7)

        ax.set_title(f"Feature {i}")
        ax.set_xlim(low, high)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def analyze_encoder_activity(input_layer, X, save_prefix: str, n_samples=2000):
    if isinstance(X, torch.Tensor):
        X_np = X.cpu().numpy()
    else:
        X_np = np.asarray(X)

    X_np = X_np[: min(n_samples, len(X_np))]
    encoded = np.stack([input_layer.encode_numpy(x) for x in X_np], axis=0)

    sum_rates = encoded.sum(axis=1)
    active_counts = (encoded > 1e-3).sum(axis=1)
    mean_per_neuron = encoded.mean(axis=0)
    dead_fraction = float(np.mean(mean_per_neuron < 1e-6))

    stats = {
        "encoded_shape": list(encoded.shape),
        "mean_total_rate": float(sum_rates.mean()),
        "std_total_rate": float(sum_rates.std()),
        "mean_active_inputs_per_state": float(active_counts.mean()),
        "std_active_inputs_per_state": float(active_counts.std()),
        "dead_neuron_fraction": dead_fraction,
    }

    fig = plt.figure(figsize=(7, 4))
    plt.hist(sum_rates, bins=40)
    plt.title("Total encoded rate per state")
    plt.xlabel("sum of rates")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(save_prefix + "_sum_rates_hist.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(7, 4))
    plt.hist(active_counts, bins=40)
    plt.title("Active encoded inputs per state")
    plt.xlabel("#active inputs")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(save_prefix + "_active_inputs_hist.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(9, 4))
    plt.bar(np.arange(len(mean_per_neuron)), mean_per_neuron)
    plt.title("Mean rate per encoded neuron")
    plt.xlabel("encoded neuron index")
    plt.ylabel("mean rate")
    plt.tight_layout()
    plt.savefig(save_prefix + "_mean_rate_per_neuron.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    return stats, encoded


def evaluate_distance_preservation(input_layer, X, n_samples=500):
    if isinstance(X, torch.Tensor):
        X_np = X.cpu().numpy()
    else:
        X_np = np.asarray(X)

    X_np = X_np[: min(n_samples, len(X_np))]
    encoded = np.stack([input_layer.encode_numpy(x) for x in X_np], axis=0)

    d_state = pairwise_distances(X_np, metric="euclidean")
    d_code = pairwise_distances(encoded, metric="euclidean")

    iu = np.triu_indices_from(d_state, k=1)
    corr, _ = spearmanr(d_state[iu], d_code[iu])
    return float(corr)


def evaluate_action_decodability(input_layer, X_train, y_train, X_val, y_val, n_train=5000, n_val=2000):
    X_train_np = X_train.cpu().numpy() if isinstance(X_train, torch.Tensor) else np.asarray(X_train)
    X_val_np = X_val.cpu().numpy() if isinstance(X_val, torch.Tensor) else np.asarray(X_val)
    y_train_np = y_train.cpu().numpy() if isinstance(y_train, torch.Tensor) else np.asarray(y_train)
    y_val_np = y_val.cpu().numpy() if isinstance(y_val, torch.Tensor) else np.asarray(y_val)

    n_train = min(n_train, len(X_train_np))
    n_val = min(n_val, len(X_val_np))

    Xtr = np.stack([input_layer.encode_numpy(x) for x in X_train_np[:n_train]], axis=0)
    Xva = np.stack([input_layer.encode_numpy(x) for x in X_val_np[:n_val]], axis=0)

    clf = LogisticRegression(max_iter=2000, multi_class="auto")
    clf.fit(Xtr, y_train_np[:n_train])
    pred = clf.predict(Xva)
    acc = accuracy_score(y_val_np[:n_val], pred)
    return float(acc)


def evaluate_poisson_spike_stats(input_layer, X, time=20, dt=0.5, n_samples=500):
    if isinstance(X, torch.Tensor):
        X_np = X.cpu().numpy()
    else:
        X_np = np.asarray(X)

    X_np = X_np[: min(n_samples, len(X_np))]

    total_spikes = []
    active_inputs = []
    per_time = []

    for x in X_np:
        rates = input_layer.step(x)
        spikes = poisson(datum=rates, time=time, dt=dt)
        spikes_np = spikes.cpu().numpy()
        total_spikes.append(float(spikes_np.sum()))
        active_inputs.append(int((spikes_np.sum(axis=0) > 0).sum()))
        per_time.append(float(spikes_np.sum(axis=1).mean()))

    return {
        "mean_total_input_spikes_per_sample": float(np.mean(total_spikes)),
        "std_total_input_spikes_per_sample": float(np.std(total_spikes)),
        "mean_active_input_neurons_per_sample": float(np.mean(active_inputs)),
        "std_active_input_neurons_per_sample": float(np.std(active_inputs)),
        "mean_spikes_per_time_step": float(np.mean(per_time)),
    }


def run_encoder_diagnostics(input_layer, X_train, y_train, X_val, y_val, diag_dir: str, time: int, dt: float):
    ensure_dirs(diag_dir)

    plot_grf_receptive_fields(input_layer, os.path.join(diag_dir, "grf_receptive_fields.png"))
    activity_stats, _ = analyze_encoder_activity(
        input_layer,
        X_train,
        save_prefix=os.path.join(diag_dir, "encoder_activity"),
        n_samples=2000,
    )
    distance_corr = evaluate_distance_preservation(input_layer, X_train, n_samples=500)
    action_dec = evaluate_action_decodability(input_layer, X_train, y_train, X_val, y_val)
    poisson_stats = evaluate_poisson_spike_stats(input_layer, X_train, time=time, dt=dt, n_samples=500)

    report = {
        "n_inpt": input_layer.n_inpt,
        "centers_per_feature": list(input_layer.centers_per_feature),
        "feature_ranges": [list(x) for x in input_layer.feature_ranges],
        "sigma_scale": float(input_layer.sigma_scale),
        "rho": float(input_layer.rho),
        "normalize_per_feature": bool(input_layer.normalize_per_feature),
        "activity_stats": activity_stats,
        "distance_preservation_spearman": distance_corr,
        "action_decodability_logreg_acc": action_dec,
        "poisson_stats": poisson_stats,
    }

    with open(os.path.join(diag_dir, "encoder_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n=== ENCODER DIAGNOSTICS ===")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


# =========================================================
# HIDDEN LAYER DIAGNOSTICS
# =========================================================
def analyze_hidden_activity(full_spike_record: torch.Tensor, save_prefix: str, active_threshold: int = 1):
    spikes_np = full_spike_record.cpu().numpy()

    total_spikes_per_sample = spikes_np.sum(axis=1)
    active_neurons_per_sample = (spikes_np >= active_threshold).sum(axis=1)
    mean_spikes_per_neuron = spikes_np.mean(axis=0)
    dead_fraction = float(np.mean(mean_spikes_per_neuron < 1e-6))

    stats = {
        "mean_total_hidden_spikes_per_sample": float(total_spikes_per_sample.mean()),
        "std_total_hidden_spikes_per_sample": float(total_spikes_per_sample.std()),
        "mean_active_hidden_neurons_per_sample": float(active_neurons_per_sample.mean()),
        "std_active_hidden_neurons_per_sample": float(active_neurons_per_sample.std()),
        "dead_hidden_neuron_fraction": dead_fraction,
        "max_mean_spikes_single_hidden_neuron": float(mean_spikes_per_neuron.max()),
        "min_mean_spikes_single_hidden_neuron": float(mean_spikes_per_neuron.min()),
    }

    fig = plt.figure(figsize=(7, 4))
    plt.hist(total_spikes_per_sample, bins=40)
    plt.title("Total hidden spikes per sample")
    plt.xlabel("sum hidden spikes")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(save_prefix + "_total_hidden_spikes_hist.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(7, 4))
    plt.hist(active_neurons_per_sample, bins=40)
    plt.title("Active hidden neurons per sample")
    plt.xlabel("#active hidden neurons")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(save_prefix + "_active_hidden_neurons_hist.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(9, 4))
    plt.bar(np.arange(len(mean_spikes_per_neuron)), mean_spikes_per_neuron)
    plt.title("Mean spikes per hidden neuron")
    plt.xlabel("hidden neuron index")
    plt.ylabel("mean spike count")
    plt.tight_layout()
    plt.savefig(save_prefix + "_mean_spikes_per_hidden_neuron.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    return stats


def analyze_hidden_assignments(assignments: torch.Tensor, n_classes: int, save_path: str):
    assignments_np = assignments.cpu().numpy()
    counts = {}
    for c in range(-1, n_classes):
        counts[str(c)] = int((assignments_np == c).sum())

    fig = plt.figure(figsize=(6, 4))
    xs = list(counts.keys())
    ys = list(counts.values())
    plt.bar(xs, ys)
    plt.title("Hidden neuron assignments")
    plt.xlabel("class label (-1 means unassigned)")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return counts


def evaluate_hidden_action_decodability(full_spike_record: torch.Tensor, actions: torch.Tensor, n_train=5000, n_val=2000):
    X_np = full_spike_record.cpu().numpy()
    y_np = actions.cpu().numpy()

    n_train = min(n_train, len(X_np))
    n_val = min(n_val, len(X_np) - n_train)
    if n_val <= 0:
        return None

    Xtr = X_np[:n_train]
    ytr = y_np[:n_train]
    Xva = X_np[n_train:n_train + n_val]
    yva = y_np[n_train:n_train + n_val]

    clf = LogisticRegression(max_iter=2000, multi_class="auto")
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xva)
    acc = accuracy_score(yva, pred)
    return float(acc)


def analyze_hidden_winners(full_spike_record: torch.Tensor, save_path: str, top_k: int = 20):
    spikes_np = full_spike_record.cpu().numpy()
    winner_idx = spikes_np.argmax(axis=1)

    unique, counts = np.unique(winner_idx, return_counts=True)
    freq = np.zeros(spikes_np.shape[1], dtype=np.int64)
    freq[unique] = counts

    top_ids = np.argsort(freq)[::-1][:top_k]

    fig = plt.figure(figsize=(9, 4))
    plt.bar(np.arange(top_k), freq[top_ids])
    plt.xticks(np.arange(top_k), top_ids, rotation=90)
    plt.title("Most frequent winner hidden neurons")
    plt.xlabel("hidden neuron index")
    plt.ylabel("#samples where neuron was top-spiking")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    stats = {
        "top_winner_neurons": top_ids.tolist(),
        "top_winner_counts": freq[top_ids].tolist(),
        "winner_entropy_proxy_num_unique_winners": int((freq > 0).sum()),
    }
    return stats


def run_hidden_diagnostics(
    full_spike_record: torch.Tensor,
    actions: torch.Tensor,
    assignments: torch.Tensor,
    n_classes: int,
    diag_dir: str,
    epoch: int,
):
    epoch_diag_dir = os.path.join(diag_dir, f"epoch_{epoch}")
    ensure_dirs(epoch_diag_dir)

    hidden_stats = analyze_hidden_activity(
        full_spike_record=full_spike_record,
        save_prefix=os.path.join(epoch_diag_dir, "hidden_activity"),
        active_threshold=1,
    )

    assignment_stats = analyze_hidden_assignments(
        assignments=assignments,
        n_classes=n_classes,
        save_path=os.path.join(epoch_diag_dir, "hidden_assignments.png"),
    )

    actions_epoch = actions[:len(full_spike_record)]
    hidden_dec = evaluate_hidden_action_decodability(
        full_spike_record=full_spike_record,
        actions=actions_epoch,
        n_train=min(3000, len(full_spike_record) // 2),
        n_val=min(1000, len(full_spike_record) // 4),
    )

    winner_stats = analyze_hidden_winners(
        full_spike_record=full_spike_record,
        save_path=os.path.join(epoch_diag_dir, "hidden_winners.png"),
        top_k=20,
    )

    epoch_report = {
        "epoch": epoch,
        "hidden_stats": hidden_stats,
        "assignment_stats": assignment_stats,
        "hidden_action_decodability": hidden_dec,
        "winner_stats": winner_stats,
    }

    with open(os.path.join(epoch_diag_dir, "hidden_report.json"), "w", encoding="utf-8") as f:
        json.dump(epoch_report, f, ensure_ascii=False, indent=2)

    print("\n=== HIDDEN LAYER DIAGNOSTICS ===")
    print(json.dumps(epoch_report, ensure_ascii=False, indent=2))
    return epoch_report


# =========================================================
# TRAIN / EVAL HELPERS
# =========================================================
def print_results(results: Dict[str, list]) -> None:
    print()
    for s in results:
        if s != "ngram" and len(results[s]) > 0:
            last = results[s][-1]
            mean = torch.mean(torch.tensor(results[s])).item()
            std = torch.std(torch.tensor(results[s])).item() if len(results[s]) > 1 else 0.0
            best = torch.max(torch.tensor(results[s])).item()
            print(f'Results for scheme "{s}": {last:.2f} (last), {mean:.2f} (mean), {std:.2f} (std.), {best:.2f} (best)')


def update_curves(curves: Dict[str, list], actions: torch.Tensor, n_classes: int, **kwargs):
    predictions = {}
    for scheme in curves:
        if scheme == "all":
            spike_record = kwargs["spike_record"]
            assignments = kwargs["assignments"]
            prediction = all_activity(spike_record, assignments, n_classes)
        elif scheme == "proportion":
            spike_record = kwargs["spike_record"]
            assignments = kwargs["assignments"]
            proportions = kwargs["proportions"]
            prediction = proportion_weighting(spike_record, assignments, proportions, n_classes)
        elif scheme == "ngram":
            prediction = torch.zeros_like(actions)
        else:
            raise NotImplementedError

        predictions[scheme] = prediction
        accuracy = torch.sum(actions.long() == prediction).float() / len(actions)
        curves[scheme].append(100 * accuracy)

    return curves, predictions


# =========================================================
# MAIN TRAINING FUNCTION
# =========================================================
def main(
    seed=0,
    n_neurons=500,
    n_train=16000,
    n_test=4000,
    inhib=0.1,
    lr=1e-6,
    lr_decay=1.0,
    time=20,
    dt=0.5,
    theta_plus=0.0,
    tc_theta_decay=1e7,
    progress_interval=100,
    update_interval=50,
    plot=False,
    train=True,
    gpu=False,
    n_epochs=15,
    w_init_hid=0.7,
    thresh=-53.0,
    refrac=0,
    centers_per_feature=(12, 12, 12, 12, 10, 10),
    sigma_scale=0.6,
    rho=1500.0,
    normalize_per_feature=True,
    n_states_cap=20000,
    q_low=0.01,
    q_high=0.99,
):
    assert n_train % update_interval == 0 and n_test % update_interval == 0, "n_train and n_test must be divisible by update_interval"

    ensure_dirs(PARAMS_PATH, SPIKES_PATH, CURVES_PATH, RESULTS_PATH, CONFUSION_PATH, DIAG_PATH)
    set_seed(seed, gpu=gpu)

    device = "cuda" if gpu and torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.set_default_tensor_type("torch.cuda.FloatTensor")

    # ------------------ Load dataset ------------------
    df = pd.read_csv(DATA_PATH)
    n_states = min(len(df), n_states_cap)
    state_arr = np.array(df["state"].apply(parse_state_column).tolist(), dtype=np.float32)[:n_states]
    action_arr = np.array(df["action"].tolist(), dtype=np.int64)[:n_states]

    X = torch.tensor(state_arr, dtype=torch.float32)
    y = torch.tensor(action_arr, dtype=torch.long)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=n_test, shuffle=True, random_state=seed, stratify=y
    )

    feature_ranges = build_feature_ranges_from_data(X_train, q_low=q_low, q_high=q_high)
    input_layer = GRFInputLayer(
        centers_per_feature=centers_per_feature,
        feature_ranges=feature_ranges,
        sigma_scale=sigma_scale,
        rho=rho,
        normalize_per_feature=normalize_per_feature,
        clip_rates=None,
        device=device,
    )

    # input_layer = OldGRFInputLayer(rho=5000.0, num_points=100, device=device)

    diag_report = run_encoder_diagnostics(
        input_layer=input_layer,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        diag_dir=DIAG_PATH,
        time=time,
        dt=dt,
    )

    # ------------------ Create/load network ------------------
    params = [
        seed, n_neurons, n_train, inhib, lr, lr_decay, time, dt,
        theta_plus, tc_theta_decay, progress_interval, update_interval,
        input_layer.n_inpt, sigma_scale, rho, w_init_hid, thresh,
    ]
    test_params = params[:3] + [n_test] + params[3:]
    model_name = "_".join([str(x) for x in params])

    if train:
        network = DiehlAndCook2015v2(
            n_inpt=input_layer.n_inpt,
            n_neurons=n_neurons,
            inh=inhib,
            dt=dt,
            norm=None,
            theta_plus=theta_plus,
            tc_theta_decay=tc_theta_decay,
            nu=[0.25 * lr, lr],
            w_init_hid=w_init_hid,
            thresh=thresh,
            refrac=refrac,
            wmin=0.0,
            wmax=3.0,
        )
    else:
        network = load(os.path.join(PARAMS_PATH, model_name + ".pt"))
        network.connections["X", "Y"].update_rule = NoOp(
            connection=network.connections["X", "Y"], nu=network.connections["X", "Y"].nu
        )
        network.layers["Y"].tc_theta_decay = torch.tensor(0.0, dtype=torch.float32)
        network.layers["Y"].theta_plus = torch.tensor(0.0, dtype=torch.float32)

    states, actions = (X_train, y_train) if train else (X_val, y_val)
    n_examples = n_train if train else n_test
    n_classes = 3

    if train:
        assignments = -torch.ones(n_neurons, dtype=torch.long)
        proportions = torch.zeros(n_neurons, n_classes)
        rates = torch.zeros(n_neurons, n_classes)
        ngram_scores = {}
    else:
        path = os.path.join(PARAMS_PATH, "_".join(["auxiliary", model_name]) + ".pt")
        assignments, proportions, rates, ngram_scores = torch.load(open(path, "rb"))

    curves = {"all": [], "proportion": [], "ngram": []}
    predictions = {scheme: torch.Tensor().long() for scheme in curves.keys()}
    best_accuracy = -1.0
    hidden_reports = []

    spikes = {}
    for layer in set(network.layers):
        spikes[layer] = Monitor(network.layers[layer], state_vars=["s"], time=int(round(time / dt)))
        network.add_monitor(spikes[layer], name=f"{layer}_spikes")

    print("\nНачало обучения.\n" if train else "\nНачало тестирования.\n")

    global_start = now()
    for epoch in range(n_epochs):
        print(f"\nEpoch {epoch + 1}/{n_epochs}")
        spike_record = torch.zeros(update_interval, int(time/dt), n_neurons)
        full_spike_record = torch.zeros(n_examples, n_neurons).long()
        start = now()

        for i in range(n_examples):
            if i % progress_interval == 0 and i != 0:
                elapsed = now() - start
                print(f"Прогресс: {i} / {n_examples} ({elapsed:.2f} секунд)")
                start = now()

            if i % update_interval == 0 and i > 0:
                if train:
                    network.connections["X", "Y"].update_rule.nu[1] *= lr_decay

                if i % len(actions) == 0:
                    current_actions = actions[-update_interval:]
                else:
                    current_actions = actions[i % len(states) - update_interval : i % len(states)]

                curves, preds = update_curves(
                    curves,
                    current_actions,
                    n_classes,
                    spike_record=spike_record,
                    assignments=assignments,
                    proportions=proportions,
                    ngram_scores=ngram_scores,
                    n=2,
                )

                print_results(curves)

                for scheme in preds:
                    if scheme != "ngram":
                        predictions[scheme] = torch.cat([predictions[scheme], preds[scheme]], -1)

                to_write = ["train"] + params if train else ["test"] + test_params
                f = "_".join([str(x) for x in to_write]) + ".pt"
                torch.save((curves, update_interval, n_examples), open(os.path.join(CURVES_PATH, f), "wb"))

                if train:
                    if any([x[-1] > best_accuracy for x in curves.values()]):
                        print("Новая лучшая точность. Сохраняю параметры сети.")
                        network.save(os.path.join(PARAMS_PATH, model_name + ".pt"))
                        path = os.path.join(PARAMS_PATH, "_".join(["auxiliary", model_name]) + ".pt")
                        torch.save((assignments, proportions, rates, ngram_scores), open(path, "wb"))
                        best_accuracy = max([x[-1] for x in curves.values()])

                    assignments, proportions, rates = assign_labels(spike_record, current_actions, n_classes, rates)

            state = states[i % len(states)]
            rates_for_state = input_layer.step(state.cpu().numpy())
            sample = poisson(datum=rates_for_state, time=time, dt=dt)
            inpts = {"X": sample}

            network.run(inputs=inpts, time=time, one_step=False)

            # y_spikes = spikes["Y"].get("s").squeeze(1).t().permute(1, 0)
            y_spikes = spikes["Y"].get("s").squeeze(1)
            spike_record[i % update_interval] = y_spikes
            full_spike_record[i] = y_spikes.sum(0).long()

            if plot:
                pass

            network.reset_state_variables()

        print(f"Прогресс: {n_examples} / {n_examples} (обработано за {(now() - start):.2f} секунд)")

        i += 1
        if i % len(actions) == 0:
            current_actions = actions[-update_interval:]
        else:
            current_actions = actions[i % len(states) - update_interval : i % len(states)]

        curves, preds = update_curves(
            curves,
            current_actions,
            n_classes,
            spike_record=spike_record,
            assignments=assignments,
            proportions=proportions,
            ngram_scores=ngram_scores,
            n=2,
        )
        print_results(curves)

        for scheme in preds:
            if scheme != "ngram":
                predictions[scheme] = torch.cat((predictions[scheme], preds[scheme]), -1)

        if train:
            network.save(os.path.join(PARAMS_PATH, model_name + "_final.pt"))
            path = os.path.join(PARAMS_PATH, "_".join(["auxiliary", model_name]) + "_final.pt")
            torch.save((assignments, proportions, rates, ngram_scores), open(path, "wb"))

        # ------------------ Hidden diagnostics per epoch ------------------
        hidden_report = run_hidden_diagnostics(
            full_spike_record=full_spike_record,
            actions=actions[:n_examples],
            assignments=assignments,
            n_classes=n_classes,
            diag_dir=DIAG_PATH,
            epoch=epoch + 1,
        )
        hidden_reports.append(hidden_report)

    print("\nОбучение завершено.\n" if train else "\nТестирование завершено.\n")
    print(f"Общее время: {now() - global_start:.2f} секунд")

    print("Средние точности:\n")
    for scheme in curves.keys():
        print(f"\t{scheme}: {float(np.mean(curves[scheme])):.2f}")

    to_write = ["train"] + params if train else ["test"] + test_params
    f = "_".join([str(x) for x in to_write]) + ".pt"
    torch.save((curves, update_interval, n_examples), open(os.path.join(CURVES_PATH, f), "wb"))

    results = [
        np.mean(curves["all"]),
        np.mean(curves["proportion"]),
        np.mean(curves["ngram"]),
        np.max(curves["all"]),
        np.max(curves["proportion"]),
        np.max(curves["ngram"]),
    ]

    result_row = params + results if train else test_params + results
    result_row = [str(x) for x in result_row]
    csv_name = "train.csv" if train else "test.csv"
    csv_path = os.path.join(RESULTS_PATH, csv_name)

    if not os.path.isfile(csv_path):
        with open(csv_path, "w", encoding="utf-8") as fcsv:
            if train:
                fcsv.write(
                    "random_seed,n_neurons,n_train,inhib,lr,lr_decay,time,timestep,theta_plus,tc_theta_decay,"
                    "progress_interval,update_interval,n_inpt,sigma_scale,rho,w_init_hid,thresh,mean_all_activity,mean_proportion_weighting,"
                    "mean_ngram,max_all_activity,max_proportion_weighting,max_ngram\n"
                )
            else:
                fcsv.write(
                    "random_seed,n_neurons,n_train,n_test,inhib,lr,lr_decay,time,timestep,theta_plus,tc_theta_decay,"
                    "progress_interval,update_interval,n_inpt,sigma_scale,rho,w_init_hid,thresh,mean_all_activity,mean_proportion_weighting,"
                    "mean_ngram,max_all_activity,max_proportion_weighting,max_ngram\n"
                )

    with open(csv_path, "a", encoding="utf-8") as fcsv:
        fcsv.write(",".join(result_row) + "\n")

    actions_for_conf = actions.clone()
    if actions_for_conf.numel() > n_examples:
        actions_for_conf = actions_for_conf[:n_examples]
    else:
        while actions_for_conf.numel() < n_examples:
            if 2 * actions_for_conf.numel() > n_examples:
                actions_for_conf = torch.cat([actions_for_conf, actions_for_conf[: n_examples - actions_for_conf.numel()]])
            else:
                actions_for_conf = torch.cat([actions_for_conf, actions_for_conf])

    confusions = {}
    for scheme in predictions:
        if scheme != "ngram":
            confusions[scheme] = confusion_matrix(actions_for_conf, predictions[scheme])

    torch.save(confusions, os.path.join(CONFUSION_PATH, f))
    torch.save(full_spike_record, os.path.join(SPIKES_PATH, f))

    summary = {
        "encoder_report": diag_report,
        "hidden_reports": hidden_reports,
        "mean_all": float(np.mean(curves["all"])),
        "mean_proportion": float(np.mean(curves["proportion"])),
        "best_all": float(np.max(curves["all"])),
        "best_proportion": float(np.max(curves["proportion"])),
        "model_name": model_name,
        "n_inpt": input_layer.n_inpt,
    }

    with open(os.path.join(DIAG_PATH, "run_summary.json"), "w", encoding="utf-8") as fsum:
        json.dump(summary, fsum, ensure_ascii=False, indent=2)

    return summary


if __name__ == "__main__":
    seed = 0
    n_neurons = 20
    n_train = 16000
    n_test = 4000
    inhib = 0.1
    lr = 1e-7
    lr_decay = 1.0
    time = 20
    dt = 0.5
    theta_plus = 0.0
    tc_theta_decay = 1e7
    progress_interval = 100
    update_interval = 50
    plot = False
    gpu = False
    train = True
    n_epochs = 15
    w_init_hid = 1.0
    thresh = -53.0
    refrac = 0

    centers_per_feature = (12, 12, 12, 12, 10, 10)
    # centers_per_feature = (24, 24, 24, 24, 20, 20)
    sigma_scale = 0.6
    rho = 500.0
    normalize_per_feature = True

    summary = main(
        seed=seed,
        n_neurons=n_neurons,
        n_train=n_train,
        n_test=n_test,
        inhib=inhib,
        lr=lr,
        lr_decay=lr_decay,
        time=time,
        dt=dt,
        theta_plus=theta_plus,
        tc_theta_decay=tc_theta_decay,
        progress_interval=progress_interval,
        update_interval=update_interval,
        plot=plot,
        train=train,
        gpu=gpu,
        n_epochs=n_epochs,
        w_init_hid=w_init_hid,
        thresh=thresh,
        refrac=refrac,
        centers_per_feature=centers_per_feature,
        sigma_scale=sigma_scale,
        rho=rho,
        normalize_per_feature=normalize_per_feature,
    )

    print("\nRUN SUMMARY:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("ok")
