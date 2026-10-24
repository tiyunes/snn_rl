# ===========================
# TEST: spiking actor for Acrobot
# Matches training dynamics from train.py
# - NO learning
# - episode_reset() preserves theta (same as train)
# - actor_reset_activity=True between steps (same as train)
# - Loads the full saved bindsnet Network directly via torch.load
#   (training uses actor.save() == torch.save(self, ...))
# ===========================

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, REPO_ROOT)  # local modified bindsnet

import ast
import random

import numpy as np
import pandas as pd
import torch
import gymnasium as gym
from tqdm.auto import tqdm

from bindsnet.encoding import poisson
from bindsnet.network.monitors import Monitor


# ---------------------------
# PATHS
# ---------------------------
DATA_PATH = os.path.join(REPO_ROOT, "data", "acrobot_state_action.csv")

# Directory with checkpoints written by train.py
CKPT_DIR = os.path.join(SCRIPT_DIR, "runs")
CKPT_PATH = os.path.join(SCRIPT_DIR, "actor.pt")


# ---------------------------
# HELPERS (same as train)
# ---------------------------
def set_seeds(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def parse_state_column(df: pd.DataFrame) -> np.ndarray:
    return np.array(df["state"].apply(ast.literal_eval).tolist(), dtype=np.float32)


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


def episode_reset(net):
    """Preserves theta; resets activity + short-term MSTDPET traces."""
    for layer in net.layers.values():
        if hasattr(layer, "v") and torch.is_tensor(layer.v):
            rest = layer.rest if hasattr(layer, "rest") else 0.0
            layer.v.fill_(float(rest))
        if hasattr(layer, "s") and torch.is_tensor(layer.s):
            layer.s.zero_()
        if hasattr(layer, "x") and torch.is_tensor(layer.x):
            layer.x.zero_()
        if hasattr(layer, "refrac_count") and torch.is_tensor(layer.refrac_count):
            layer.refrac_count.zero_()
    for conn in net.connections.values():
        rule = conn.update_rule
        for attr in ["eligibility_trace", "p_plus", "p_minus", "eligibility"]:
            if hasattr(rule, attr) and getattr(rule, attr) is not None:
                try:
                    getattr(rule, attr).zero_()
                except Exception:
                    pass
    for monitor in net.monitors.values():
        monitor.reset_state_variables()


def reset_activity_only(net):
    """Within-step activity reset (preserves theta and traces)."""
    for layer in net.layers.values():
        if hasattr(layer, "v") and torch.is_tensor(layer.v):
            rest = layer.rest if hasattr(layer, "rest") else 0.0
            layer.v.fill_(float(rest))
        if hasattr(layer, "s") and torch.is_tensor(layer.s):
            layer.s.zero_()
        if hasattr(layer, "x") and torch.is_tensor(layer.x):
            layer.x.zero_()
        if hasattr(layer, "refrac_count") and torch.is_tensor(layer.refrac_count):
            layer.refrac_count.zero_()
    for monitor in net.monitors.values():
        monitor.reset_state_variables()


def e_greedy_action(action_ind: int, epsilon=0.0) -> int:
    prob = (epsilon / 3.0) * np.ones(3)
    prob[action_ind] = (1.0 - epsilon) + (epsilon / 3.0)
    return int(np.random.choice(3, 1, p=prob))


# ---------------------------
# ACTOR ENCODER (same as train: compact GRF, 68 inputs)
# ---------------------------
class GRFInputLayer:
    def __init__(
        self,
        centers_per_feature=(12, 12, 12, 12, 10, 10),
        feature_ranges=None,
        sigma_scale: float = 0.6,
        rho: float = 500.0,
        normalize_per_feature: bool = True,
        device: str = "cpu",
    ):
        self.num_features = 6
        self.centers_per_feature = tuple(centers_per_feature)
        self.sigma_scale = float(sigma_scale)
        self.rho = float(rho)
        self.normalize_per_feature = bool(normalize_per_feature)
        self.device = device

        if feature_ranges is None:
            feature_ranges = [
                (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0),
                (-12.567, 12.567), (-28.274, 28.274),
            ]
        self.feature_ranges = feature_ranges
        self.centers = []
        self.sigmas = []
        for i in range(self.num_features):
            low, high = self.feature_ranges[i]
            n_c = self.centers_per_feature[i]
            c = np.linspace(low, high, n_c, dtype=np.float32)
            spacing = (c[1] - c[0]) if n_c > 1 else max(high - low, 1.0)
            self.centers.append(c)
            self.sigmas.append(np.full_like(c, max(self.sigma_scale * spacing, 1e-6)))
        self.n_inpt = int(sum(self.centers_per_feature))

    def encode_numpy(self, state):
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        outputs = []
        for i in range(self.num_features):
            resp = np.exp(-0.5 * ((state[i] - self.centers[i]) / self.sigmas[i]) ** 2)
            if self.normalize_per_feature:
                resp = resp / (resp.sum() + 1e-8)
            outputs.append(resp)
        return (np.concatenate(outputs, axis=0) * self.rho).astype(np.float32)

    def rates(self, state):
        return torch.tensor(self.encode_numpy(state), dtype=torch.float32, device=self.device)


def build_grf_spike_train(encoder: GRFInputLayer, state, sim_time: float, dt: float) -> torch.Tensor:
    rates = encoder.rates(np.asarray(state, dtype=np.float32))
    spk = poisson(datum=rates, time=sim_time, dt=dt)
    if spk.ndim == 2:
        spk = spk.unsqueeze(1)
    return spk.float()


# ---------------------------
# CHECKPOINT LOADING
# ---------------------------
def load_actor_network(ckpt_path: str, device="cpu"):
    """Training saves the actor via bindsnet's Network.save() == torch.save(self,...).

    So torch.load returns the full Network object (with H weights, Y weights, theta,
    refrac_count, etc.). We strip the pickled monitors and re-attach fresh ones below.
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    obj = torch.load(ckpt_path, map_location=device)
    if not hasattr(obj, "layers") or not hasattr(obj, "connections"):
        raise ValueError(
            f"Loaded object is not a bindsnet Network: {type(obj)}. "
            f"This test script expects checkpoints saved via actor.save() in train.py."
        )
    # Drop any pickled monitors so we can re-attach fresh ones tied to the test loop.
    if hasattr(obj, "monitors"):
        obj.monitors = {}
    print(f"[CKPT] Loaded actor network from: {ckpt_path}")
    print(f"[CKPT] layers: {list(obj.layers.keys())}  "
          f"hid={obj.layers['H'].n}  out={obj.layers['Y'].n}")
    return obj


# ---------------------------
# TEST
# ---------------------------
def test_actor(
    ckpt_path: str = CKPT_PATH,
    seed: int = 0,
    n_episodes: int = 20,
    max_steps: int = 1000,
    time_sec: float = 20.0,
    dt: float = 0.5,
    epsilon: float = 0.0,
    render: bool = True,
    actor_reset_activity: bool = True,
    centers_per_feature=(12, 12, 12, 12, 10, 10),
    sigma_scale: float = 0.6,
    rho: float = 500.0,
    normalize_per_feature: bool = True,
    q_low: float = 0.01,
    q_high: float = 0.99,
):
    set_seeds(seed)

    timesteps = int(round(time_sec / dt))
    sim_time = time_sec

    # Encoder with the same feature_ranges and parameters as in train.py
    df = pd.read_csv(DATA_PATH)
    states_np = parse_state_column(df)
    feature_ranges = build_feature_ranges_from_data(states_np, q_low=q_low, q_high=q_high)

    actor_enc = GRFInputLayer(
        centers_per_feature=centers_per_feature,
        feature_ranges=feature_ranges,
        sigma_scale=sigma_scale,
        rho=rho,
        normalize_per_feature=normalize_per_feature,
    )
    print(f"[ENC] n_inpt={actor_enc.n_inpt}, "
          f"feature_ranges[4]={feature_ranges[4]}, [5]={feature_ranges[5]}")

    # Load actor (full bindsnet Network with H weights, theta, etc.)
    actor_net = load_actor_network(ckpt_path)
    if actor_net.layers["X"].n != actor_enc.n_inpt:
        raise ValueError(
            f"Encoder/actor input mismatch: encoder n_inpt={actor_enc.n_inpt}, "
            f"actor X layer n={actor_net.layers['X'].n}"
        )

    # Fresh monitors (buffer = timesteps)
    actor_spikes = {}
    for lname in actor_net.layers:
        actor_spikes[lname] = Monitor(actor_net.layers[lname], state_vars=["s"], time=timesteps)
        actor_net.add_monitor(actor_spikes[lname], name=f"{lname}_spikes")

    # Env
    env = gym.make("Acrobot-v1", render_mode=("human" if render else None), max_episode_steps=max_steps)
    env.reset(seed=seed)

    lens = []
    terminations = []

    for ep in tqdm(range(n_episodes), desc="TEST"):
        state, _ = env.reset(seed=seed + ep)
        episode_reset(actor_net)

        terminated_ep = False

        for ti in range(1, max_steps + 1):
            if actor_reset_activity:
                reset_activity_only(actor_net)

            spk_X = build_grf_spike_train(actor_enc, state=state, sim_time=sim_time, dt=dt)

            actor_net.run(
                inputs={"X": spk_X},
                time=sim_time,
                learning=False,
                reward=0.0,
                step=ti - 1,
                one_step=False,
            )

            y_counts = actor_spikes["Y"].get("s").sum(dim=0)[0].detach()
            action_pred = int(torch.argmax(y_counts).item())
            action = e_greedy_action(action_pred, epsilon=epsilon)

            next_state, _, terminated, truncated, _ = env.step(int(action))
            terminated_ep = bool(terminated_ep or terminated)

            state = next_state

            if terminated or truncated:
                break

        lens.append(ti)
        terminations.append(int(terminated_ep))
        print(f"EP {ep:03d}: steps={ti:4d} | success={int(terminated_ep)}")

    env.close()

    lens_arr = np.array(lens)
    print("\n========== SUMMARY ==========")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Episodes  : {n_episodes}")
    print(f"Success   : {sum(terminations)}/{n_episodes} = {100.0 * sum(terminations) / n_episodes:.1f}%")
    print(f"Mean steps: {lens_arr.mean():.2f}")
    print(f"Min  steps: {lens_arr.min()}")
    print(f"Max  steps: {lens_arr.max()}")


if __name__ == "__main__":
    # Change to actor_ep<N>.pt or actor_final.pt to test a different checkpoint
    # ckpt_path = os.path.join(CKPT_DIR, "actor_best.pt")

    test_actor(
        ckpt_path=CKPT_PATH,
        seed=0,
        n_episodes=100,
        max_steps=1000,
        time_sec=20.0,
        dt=0.5,
        epsilon=0.0,
        render=True,
        actor_reset_activity=True,
        centers_per_feature=(12, 12, 12, 12, 10, 10),
        sigma_scale=0.6,
        rho=500.0,
        normalize_per_feature=True,
    )
