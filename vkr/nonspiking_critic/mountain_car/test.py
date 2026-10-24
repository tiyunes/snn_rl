# ===========================
# TEST: MountainCar SNN actor (compact GRF, 500 hidden, linear Fourier critic during train)
# Loads actor checkpoints from train.py:
#   actor_ep_<N>.pt   (saved via bindsnet network.save())
#   critic_ep_<N>.npy (Fourier critic weights — not used here)
#   summary_ep_<N>.json
#
# - NO learning
# - safe_reset() between episodes  (matches training)
# - no activity reset between steps (matches training; state carries forward inside an episode)
# ===========================

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, REPO_ROOT)  # local modified bindsnet

import glob
import json
import random
from typing import Optional

import numpy as np
import pandas as pd
import torch
import gymnasium as gym
from tqdm.auto import tqdm

from bindsnet.encoding import poisson
from bindsnet.network import load as bn_load
from bindsnet.network.monitors import Monitor


# ---------------------------
# PATHS
# ---------------------------
DATA_PATH = os.path.join(REPO_ROOT, "data", "mountaincar_state_action.csv")

# Directory with training checkpoints (used when CKPT_PATH is None).
SAVE_DIR = os.path.join(SCRIPT_DIR, "runs")

# Set to None to auto-pick best by `best_ckpt_metric` (see test_actor args below).
CKPT_PATH = os.path.join(SCRIPT_DIR, "actor.pt")


# ---------------------------
# HELPERS
# ---------------------------
def set_seeds(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def load_states_from_csv(csv_path: str) -> np.ndarray:
    df = pd.read_csv(csv_path)
    return df[["position", "velocity"]].to_numpy(dtype=np.float32)


def build_feature_ranges_from_data(X_np: np.ndarray, q_low=0.01, q_high=0.99):
    ranges = []
    for j in range(X_np.shape[1]):
        low = float(np.quantile(X_np[:, j], q_low))
        high = float(np.quantile(X_np[:, j], q_high))
        if abs(high - low) < 1e-8:
            high = low + 1e-3
        margin = 0.05 * (high - low)
        ranges.append((low - margin, high + margin))
    return ranges


def safe_reset(net):
    """Match training behavior: full reset including theta + MSTDPET trace zero-out."""
    net.reset_state_variables()
    for conn in net.connections.values():
        rule = conn.update_rule
        if hasattr(rule, "reset_state_variables"):
            try:
                rule.reset_state_variables()
            except TypeError:
                pass
        for attr in ["eligibility_trace", "p_plus", "p_minus", "eligibility"]:
            if hasattr(rule, attr) and getattr(rule, attr) is not None:
                try:
                    getattr(rule, attr).zero_()
                except Exception:
                    pass


def e_greedy_action(action_ind: int, epsilon=0.0, n_actions=3) -> int:
    probs = np.full(n_actions, epsilon / n_actions, dtype=np.float64)
    probs[action_ind] += (1.0 - epsilon)
    return int(np.random.choice(np.arange(n_actions), p=probs))


def get_spike_counts(spikes: dict, layer_name: str) -> torch.Tensor:
    s = spikes[layer_name].get("s")
    return s.sum(dim=0)[0].detach().clone()


# ---------------------------
# ENCODER (compact GRF, matches train.py)
# ---------------------------
class InputLayerMountainCar:
    def __init__(
        self,
        centers_per_feature=(30, 20),
        feature_ranges=None,
        sigma_scale=0.6,
        rho=1500.0,
        normalize_per_feature=True,
        device="cpu",
    ):
        self.num_features = 2
        self.centers_per_feature = tuple(centers_per_feature)
        self.sigma_scale = float(sigma_scale)
        self.rho = float(rho)
        self.normalize_per_feature = bool(normalize_per_feature)
        self.device = device

        if feature_ranges is None:
            feature_ranges = [(-1.2, 0.6), (-0.07, 0.07)]
        self.feature_ranges = feature_ranges

        self.centers = []
        self.sigmas = []
        for i in range(self.num_features):
            low, high = self.feature_ranges[i]
            n_c = self.centers_per_feature[i]
            c = np.linspace(low, high, n_c, dtype=np.float32)
            spacing = (c[1] - c[0]) if n_c > 1 else max(high - low, 1.0)
            sigma = max(self.sigma_scale * spacing, 1e-6)
            self.centers.append(c)
            self.sigmas.append(np.full_like(c, sigma, dtype=np.float32))

        self.n_inpt = int(sum(self.centers_per_feature))

    def encode_numpy(self, state):
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        outputs = []
        for i in range(self.num_features):
            x = state[i]
            c = self.centers[i]
            s = self.sigmas[i]
            resp = np.exp(-0.5 * ((x - c) / s) ** 2)
            if self.normalize_per_feature:
                resp = resp / (resp.sum() + 1e-8)
            outputs.append(resp)
        return np.concatenate(outputs, axis=0) * self.rho

    def rates(self, state):
        return torch.tensor(
            self.encode_numpy(state), dtype=torch.float32, device=self.device
        )


# ---------------------------
# CHECKPOINT SELECTION
# ---------------------------
def find_best_checkpoint(save_dir: str, metric: str = "mean_reward_100") -> str:
    """Scan summary_ep_*.json files and return path to actor_ep_<N>.pt
    with the largest value of `metric`.

    Falls back to the latest actor_ep_*.pt if no summaries are found.
    metric options: 'mean_reward_100', 'success_rate_100', 'mean_final_position_100'.
    """
    summary_files = glob.glob(os.path.join(save_dir, "summary_ep_*.json"))
    if not summary_files:
        actor_files = glob.glob(os.path.join(save_dir, "actor_ep_*.pt"))
        if not actor_files:
            raise FileNotFoundError(f"No checkpoints found in {save_dir}")
        ep_files = []
        for f in actor_files:
            try:
                ep = int(os.path.basename(f).replace("actor_ep_", "").replace(".pt", ""))
                ep_files.append((ep, f))
            except ValueError:
                pass
        _, best_path = max(ep_files, key=lambda x: x[0])
        print(f"[CKPT] No summaries. Using latest checkpoint: {best_path}")
        return best_path

    best_val = -1e18
    best_ep = None
    for sf in summary_files:
        try:
            ep = int(os.path.basename(sf).replace("summary_ep_", "").replace(".json", ""))
            with open(sf, "r", encoding="utf-8") as f:
                data = json.load(f)
            val = data.get(metric)
            if val is None:
                continue
            if float(val) > best_val:
                best_val = float(val)
                best_ep = ep
        except Exception:
            pass

    if best_ep is None:
        raise RuntimeError(f"Could not parse any summary files in {save_dir}")

    best_path = os.path.join(save_dir, f"actor_ep_{best_ep}.pt")
    if not os.path.exists(best_path):
        raise FileNotFoundError(f"Best ep={best_ep} but {best_path} is missing")

    print(f"[CKPT] Best by '{metric}'={best_val:.4f} at ep={best_ep}: {best_path}")
    return best_path


# ---------------------------
# TEST
# ---------------------------
def test_actor(
    ckpt_path: Optional[str] = None,
    seed: int = 0,
    n_episodes: int = 100,
    max_steps: int = 1000,
    time_sec: float = 20.0,
    dt: float = 0.5,

    # encoder (matches training)
    centers_per_feature=(30, 20),
    sigma_scale: float = 0.6,
    rho: float = 1500.0,
    is_input_normalize: bool = True,
    q_low: float = 0.01,
    q_high: float = 0.99,

    epsilon: float = 0.0,
    render: bool = True,
    best_ckpt_metric: str = "mean_reward_100",
):
    set_seeds(seed)
    device = torch.device("cpu")

    if ckpt_path is None:
        ckpt_path = find_best_checkpoint(SAVE_DIR, metric=best_ckpt_metric)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    timesteps = int(round(time_sec / dt))
    sim_time = time_sec

    # ------ encoder ------
    states_np = load_states_from_csv(DATA_PATH)
    feature_ranges = build_feature_ranges_from_data(states_np, q_low=q_low, q_high=q_high)

    actor_enc = InputLayerMountainCar(
        centers_per_feature=centers_per_feature,
        feature_ranges=feature_ranges,
        sigma_scale=sigma_scale,
        rho=rho,
        normalize_per_feature=is_input_normalize,
        device=device,
    )
    print(
        f"[Encoder] compact-GRF n_inpt={actor_enc.n_inpt}  "
        f"centers={centers_per_feature}  sigma_scale={sigma_scale}  rho={rho}  "
        f"normalize_per_feature={is_input_normalize}"
    )

    # ------ load network (bindsnet network.save format) ------
    net = bn_load(ckpt_path)
    net.to(device)
    print(f"[Network] Loaded from {ckpt_path}")
    print(f"[Network] hidden={net.n_neurons_hid} output={net.n_neurons_out}")

    if ("X", "H") in net.connections:
        xh = net.connections[("X", "H")].w
        if xh.shape[0] != actor_enc.n_inpt:
            raise ValueError(
                f"Input dim mismatch: encoder n_inpt={actor_enc.n_inpt}, "
                f"network expects {xh.shape[0]}"
            )

    spikes = {}
    for layer in net.layers:
        spikes[layer] = Monitor(net.layers[layer], state_vars=["s"], time=timesteps)
        net.add_monitor(spikes[layer], name=f"{layer}_spikes")

    # ------ env ------
    env = gym.make(
        "MountainCar-v0",
        render_mode=("human" if render else None),
        max_episode_steps=max_steps,
    )
    env.reset(seed=seed)

    lens, successes, final_positions, rewards = [], [], [], []

    for ep in tqdm(range(n_episodes), desc="TEST"):
        safe_reset(net)
        state, _ = env.reset(seed=seed + ep)

        ep_reward = 0.0
        success_flag = False

        for ti in range(1, max_steps + 1):
            if isinstance(state, tuple):
                state = state[0]

            rates_X = actor_enc.rates(state)
            spk_X = poisson(datum=rates_X, time=int(time_sec), dt=dt)

            net.run(
                inputs={"X": spk_X},
                time=sim_time,
                learning=False,
                reward=0.0,
                step=ti - 1,
                one_step=False,
            )

            y_counts = get_spike_counts(spikes, "Y")
            action_pred = int(torch.argmax(y_counts).item())
            action = e_greedy_action(action_pred, epsilon=epsilon, n_actions=3)

            state, _, terminated, truncated, _ = env.step(int(action))
            ep_reward += -1.0

            if terminated:
                success_flag = True
                break
            if truncated:
                break

        lens.append(ti)
        successes.append(int(success_flag))
        final_positions.append(float(state[0]))
        rewards.append(ep_reward)

        print(
            f"EP {ep:03d}: steps={ti:4d} | "
            f"final_pos={state[0]:+.4f} | "
            f"success={int(success_flag)}"
        )

    env.close()

    lens_arr = np.array(lens)
    pos_arr = np.array(final_positions)
    rew_arr = np.array(rewards)

    print("\n========== SUMMARY ==========")
    print(f"Checkpoint : {ckpt_path}")
    print(f"Episodes   : {n_episodes}")
    print(f"Success    : {sum(successes)}/{n_episodes} = {100.0 * sum(successes) / max(n_episodes, 1):.1f}%")
    print(f"Mean steps : {lens_arr.mean():.2f}")
    print(f"Min  steps : {lens_arr.min()}")
    print(f"Max  steps : {lens_arr.max()}")
    print(f"Mean reward: {rew_arr.mean():.2f}")
    print(f"Mean final pos: {pos_arr.mean():.4f}")
    print(f"Max  final pos: {pos_arr.max():.4f}")


if __name__ == "__main__":
    test_actor(
        ckpt_path=CKPT_PATH,        # None → auto-pick best by best_ckpt_metric
        seed=0,
        n_episodes=100,
        max_steps=1000,
        time_sec=20.0,
        dt=0.5,

        centers_per_feature=(30, 20),
        sigma_scale=0.6,
        rho=1500.0,
        is_input_normalize=True,
        q_low=0.01,
        q_high=0.99,

        epsilon=0.0,
        render=True,
        best_ckpt_metric="mean_reward_100",
    )
