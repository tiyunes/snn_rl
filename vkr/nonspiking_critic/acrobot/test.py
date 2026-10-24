# ===========================
# TEST: spiking actor for Acrobot
# Matches training dynamics from train.py
# - NO learning
# - episode_reset() preserves theta (same as train)
# - actor_reset_activity=True between steps (same as train)
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
from bindsnet.learning import NoOp
from bindsnet.network.monitors import Monitor
from bindsnet.models import Diehl2LayersWTAOutput

# ---------------------------
# PATHS
# ---------------------------
DATA_PATH = os.path.join(REPO_ROOT, "data", "acrobot_state_action.csv")
HIDDEN_LAYER_PATH = os.path.join(REPO_ROOT, "hidden_layers", "acrobot_20n", "hidden_layer.pt")
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
# ACTOR ENCODER (same as train)
# ---------------------------
class InputLayer:
    def __init__(
        self,
        centers_per_feature=(24, 24, 24, 24, 20, 20),
        feature_ranges=None,
        sigma_scale=0.6,
        rho=500.0,
        normalize_per_feature=True,
        device="cpu",
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
            low, high = feature_ranges[i]
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
        return torch.tensor(self.encode_numpy(state), dtype=torch.float32, device=self.device)


def build_grf_spike_train(encoder: InputLayer, state, sim_time: float, dt: float) -> torch.Tensor:
    rates = encoder.rates(np.asarray(state, dtype=np.float32))
    spk = poisson(datum=rates, time=sim_time, dt=dt)
    if spk.ndim == 2:
        spk = spk.unsqueeze(1)
    return spk.float()


# ---------------------------
# ACTOR NET (same params as train)
# ---------------------------
def build_actor_net(
    seed=0,
    dt=0.5,
    n_inpt=68,
    n_actions=3,
    n_neurons_hid=20,
    inhib_hid=0.1,
    inhib_out=0.1,
    lr_hid=1e-7,
    lr_out=1e-4,
    w_init_hid=0.7,
    w_init_out=0.5,
    thresh_hid=-53.0,
    thresh_out=-64.0,
    refrac_hid=0,
    refrac_out=0,
    tc_trace_input=20.0,
    tc_decay_hid=15.0,
    tc_decay_out=15.0,
    tc_trace_hid=10.0,
    tc_trace_out=10.0,
    tc_plus_out=10.0,
    tc_minus_out=10.0,
    tc_e_trace_out=15.0,
    wmin_out=0.0,
    wmax_out=1.0,
):
    set_seeds(seed)

    net = Diehl2LayersWTAOutput(
        n_inpt=n_inpt,
        n_neurons_out=n_actions,
        n_neurons_hid=n_neurons_hid,
        inh_hid=inhib_hid,
        inh_out=inhib_out,
        dt=dt,
        nu_hid=[0.25 * lr_hid, lr_hid],
        nu_out=[0.25 * lr_out, lr_out],
        hidden_layer_path=HIDDEN_LAYER_PATH,
        w_init_hid=w_init_hid,
        w_init_out=w_init_out,
        thresh_hid=thresh_hid,
        thresh_out=thresh_out,
        refrac_hid=refrac_hid,
        refrac_out=refrac_out,
        one_spike_hid=False,
        one_spike_out=True,
        tc_trace_input=tc_trace_input,
        tc_decay_hid=tc_decay_hid,
        tc_decay_out=tc_decay_out,
        tc_trace_hid=tc_trace_hid,
        tc_trace_out=tc_trace_out,
        tc_plus_out=tc_plus_out,
        tc_minus_out=tc_minus_out,
        tc_e_trace_out=tc_e_trace_out,
        wmin_out=wmin_out,
        wmax_out=wmax_out,
    )

    if ("X", "H") in net.connections:
        net.connections[("X", "H")].update_rule = NoOp(
            connection=net.connections[("X", "H")],
            nu=net.connections[("X", "H")].nu,
        )

    return net


# ---------------------------
# CHECKPOINT LOADING
# ---------------------------
def load_actor_from_checkpoint(path: str, actor_net, device="cpu"):
    obj = torch.load(path, map_location=device)

    if hasattr(obj, "connections"):
        # BindsNET Network.save() format — full network object
        for key, conn in obj.connections.items():
            if key in actor_net.connections:
                actor_net.connections[key].w.data.copy_(
                    conn.w.data.to(actor_net.connections[key].w.device)
                )
        for lname, layer in obj.layers.items():
            if lname in actor_net.layers:
                tgt = actor_net.layers[lname]
                if hasattr(layer, "theta") and torch.is_tensor(layer.theta):
                    tgt.theta.data.copy_(layer.theta.data.to(tgt.theta.device))
        extra = {}
    else:
        # Old dict format with actor_connections / actor_layers keys
        ckpt = obj
        for key_str, data in ckpt["actor_connections"].items():
            for conn_key, conn in actor_net.connections.items():
                if str(conn_key) == key_str:
                    conn.w.data.copy_(data["w"].to(conn.w.device))
                    break
        for lname, layer_data in ckpt.get("actor_layers", {}).items():
            if lname in actor_net.layers:
                layer = actor_net.layers[lname]
                for attr, t in layer_data.items():
                    if hasattr(layer, attr) and torch.is_tensor(getattr(layer, attr)):
                        getattr(layer, attr).data.copy_(t.to(getattr(layer, attr).device))
        extra = ckpt.get("extra", {})

    print(f"[CKPT] Loaded actor from: {path}")
    if extra:
        print(f"[CKPT] extra: {extra}")
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
):
    set_seeds(seed)

    timesteps = int(round(time_sec / dt))
    sim_time = time_sec

    # build encoder with same feature_ranges as training
    df = pd.read_csv(DATA_PATH)
    states_np = parse_state_column(df)
    feature_ranges = build_feature_ranges_from_data(states_np, q_low=0.01, q_high=0.99)

    actor_enc = InputLayer(
        centers_per_feature=(12, 12, 12, 12, 10, 10),
        feature_ranges=feature_ranges,
        sigma_scale=0.6,
        rho=500.0,
        normalize_per_feature=True,
    )
    print(f"[ENC] n_inpt={actor_enc.n_inpt}, feature_ranges[4]={feature_ranges[4]}, [5]={feature_ranges[5]}")

    # build net
    actor_net = build_actor_net(seed=seed, dt=dt, n_inpt=actor_enc.n_inpt)

    # monitors
    actor_spikes = {}
    for layer in actor_net.layers:
        actor_spikes[layer] = Monitor(actor_net.layers[layer], state_vars=["s"], time=timesteps)
        actor_net.add_monitor(actor_spikes[layer], name=f"{layer}_spikes")

    # load checkpoint (actor weights only)
    load_actor_from_checkpoint(ckpt_path, actor_net)

    # env
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
    )
