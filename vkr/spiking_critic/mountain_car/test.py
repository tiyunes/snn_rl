# ===========================
# TEST: MountainCar SNN actor (InputLayer 200-input, 1000 hidden neurons)
#
# Loads a checkpoint produced by train.py
# (custom dict format with actor_connections / actor_layers / critic_connections / critic_layers).
#
# - NO learning
# - episode_reset() between episodes
# - reset_activity_only() before each step (matches actor_reset_activity=True in training)
# ===========================

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, REPO_ROOT)  # local modified bindsnet

import glob
import random
from typing import Optional

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
DATA_PATH = os.path.join(REPO_ROOT, "data", "mountaincar_state_action.csv")
HIDDEN_LAYER_PATH = os.path.join(REPO_ROOT, "hidden_layers", "mountain_car_1000n", "hidden_layer.pt")

# Directory with checkpoints written by train.py
SAVE_DIR = os.path.join(SCRIPT_DIR, "runs")
CKPT_PATH = os.path.join(SCRIPT_DIR, "checkpoint.pt")


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


def e_greedy_action(action_ind: int, epsilon=0.0, n_actions=3) -> int:
    probs = np.full(n_actions, epsilon / n_actions, dtype=np.float64)
    probs[action_ind] += (1.0 - epsilon)
    return int(np.random.choice(np.arange(n_actions), p=probs))


def get_spike_counts(spikes: dict, layer_name: str) -> torch.Tensor:
    s = spikes[layer_name].get("s")
    return s.sum(dim=0)[0].detach().clone()


def reset_activity_only(net):
    """Reset membrane/spike activity but preserve theta."""
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


def episode_reset(net):
    """Activity + short-term MSTDPET traces zeroed; theta preserved."""
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


# ---------------------------
# ENCODER (cartpole-style InputLayer, same as training)
# ---------------------------
class InputLayerMountainCar:
    def __init__(
        self,
        num_points: int = 100,
        rho: float = 20000.0,
        pos_range=(-1.2, 0.6),
        vel_range=(-0.07, 0.07),
        device: str = "cpu",
    ):
        self.num_features = 2
        self.num_points = int(num_points)
        self.rho = float(rho)
        self.device = device

        pos_values = np.linspace(pos_range[0], pos_range[1], self.num_points, dtype=np.float32)
        vel_values = np.linspace(vel_range[0], vel_range[1], self.num_points, dtype=np.float32)

        h = np.zeros(self.num_features, dtype=np.float32)
        h[0] = pos_values[1] - pos_values[0]
        h[1] = vel_values[1] - vel_values[0]
        self.h = h.reshape(self.num_features, 1)

        self.values = np.vstack((pos_values, vel_values)).astype(np.float32)
        self.n_inpt = self.num_features * self.num_points

    def encode_numpy(self, state):
        state = np.asarray(state, dtype=np.float32).reshape(self.num_features, 1)
        diff = (self.values - state) ** 2
        exp = np.exp(-0.5 * diff / (self.h ** 2))
        return (self.rho * exp.flatten()).astype(np.float32)

    def rates(self, state):
        return torch.tensor(self.encode_numpy(state), dtype=torch.float32, device=self.device)


# ---------------------------
# ACTOR (same architecture as training)
# ---------------------------
def build_actor_net(
    seed=0,
    dt=0.5,
    n_inpt=200,
    n_actions=3,
    n_neurons_hid=1000,
    inhib_hid=0.1,
    inhib_out=0.1,
    lr_hid=1e-5,
    lr_out=1e-4,
    w_init_hid=0.5,
    w_init_out=0.5,
    thresh_hid=-56.0,
    thresh_out=-64.0,
    refrac_hid=5,
    refrac_out=0,
    one_spike_hid=False,
    one_spike_out=True,
    tc_trace_input=20.0,
    tc_decay_hid=15.0,
    tc_decay_out=15.0,
    tc_trace_hid=10.0,
    tc_trace_out=10.0,
    tc_plus_out=10.0,
    tc_minus_out=10.0,
    tc_e_trace_out=15.0,
    wmin_out=0.0,
    wmax_out=2.0,
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
        one_spike_hid=one_spike_hid,
        one_spike_out=one_spike_out,
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
# CHECKPOINT LOADING (matches training script's save_checkpoint format)
# ---------------------------
def load_actor_from_checkpoint(actor_net, ckpt_path: str) -> dict:
    """Copy weights and theta from a training checkpoint into the actor.

    The training checkpoint is a dict with:
      - actor_connections[str(key)]['w']
      - actor_layers[lname]['theta']
      - extra (metadata)
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    n_w = 0
    for key, conn in actor_net.connections.items():
        k = str(key)
        if k in ckpt.get("actor_connections", {}):
            conn.w.data.copy_(ckpt["actor_connections"][k]["w"])
            n_w += 1

    n_th = 0
    for lname, layer in actor_net.layers.items():
        if lname in ckpt.get("actor_layers", {}) and hasattr(layer, "theta"):
            layer.theta.data.copy_(ckpt["actor_layers"][lname]["theta"])
            n_th += 1

    extra = ckpt.get("extra", {})
    print(f"[CKPT] Loaded {ckpt_path}")
    print(f"[CKPT]   saved at:  {ckpt.get('time', '?')}")
    print(f"[CKPT]   weights:   {n_w} connections restored")
    print(f"[CKPT]   theta:     {n_th} layers restored")
    if "episode" in extra:
        print(f"[CKPT]   episode:   {extra['episode']}")
    if "mean_ep_len_100" in extra:
        print(f"[CKPT]   mean_len(100): {extra['mean_ep_len_100']:.2f}")
    return extra


def find_best_ckpt_in_dir(save_dir: str) -> str:
    """Pick the best ckpt_ep*_best_ep_len_*.pt by smallest ep_len, else ckpt_best.pt, else latest."""
    best_files = glob.glob(os.path.join(save_dir, "ckpt_ep?????_best_ep_len_*.pt"))
    if best_files:
        def _ep_len(p):
            base = os.path.basename(p)
            try:
                return int(base.split("_best_ep_len_")[1].replace(".pt", ""))
            except (IndexError, ValueError):
                return 10**9
        path = min(best_files, key=_ep_len)
        print(f"[CKPT] Auto-selected best-ep-len ckpt: {path}")
        return path

    fallback = os.path.join(save_dir, "ckpt_best.pt")
    if os.path.exists(fallback):
        print(f"[CKPT] Falling back to {fallback}")
        return fallback

    periodic = sorted(glob.glob(os.path.join(save_dir, "ckpt_ep?????.pt")))
    if periodic:
        print(f"[CKPT] Falling back to latest periodic: {periodic[-1]}")
        return periodic[-1]

    raise FileNotFoundError(f"No checkpoints found in {save_dir}")


# ---------------------------
# TEST
# ---------------------------
def test_actor(
    ckpt_path: Optional[str] = None,
    seed: int = 0,
    n_episodes: int = 100,
    max_steps: int = 1000,
    time_sec: float = 40.0,
    dt: float = 0.5,

    # encoder (matches training)
    encoder_num_points: int = 100,
    encoder_rho: float = 20000.0,
    encoder_pos_range=(-1.2, 0.6),
    encoder_vel_range=(-0.07, 0.07),

    # actor architecture (matches training)
    n_neurons_hid: int = 1000,
    inhib_hid: float = 0.1,
    inhib_out: float = 0.1,
    thresh_hid: float = -56.0,
    thresh_out: float = -63.0,
    refrac_hid: int = 5,
    refrac_out: int = 0,

    # inference settings
    actor_reset_activity: bool = True,
    epsilon: float = 0.0,
    render: bool = True,
):
    set_seeds(seed)
    device = torch.device("cpu")

    if ckpt_path is None:
        ckpt_path = find_best_ckpt_in_dir(SAVE_DIR)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    timesteps = int(round(time_sec / dt))
    sim_time = time_sec

    # ------ encoder ------
    actor_enc = InputLayerMountainCar(
        num_points=encoder_num_points,
        rho=encoder_rho,
        pos_range=encoder_pos_range,
        vel_range=encoder_vel_range,
        device=device,
    )
    print(
        f"[Encoder] InputLayer  n_inpt={actor_enc.n_inpt}  rho={encoder_rho}  "
        f"num_points={encoder_num_points}"
    )

    # ------ actor ------
    actor_net = build_actor_net(
        seed=seed,
        dt=dt,
        n_inpt=actor_enc.n_inpt,
        n_actions=3,
        n_neurons_hid=n_neurons_hid,
        inhib_hid=inhib_hid,
        inhib_out=inhib_out,
        thresh_hid=thresh_hid,
        thresh_out=thresh_out,
        refrac_hid=refrac_hid,
        refrac_out=refrac_out,
    )
    actor_net.to(device)

    if ("X", "H") in actor_net.connections:
        xh = actor_net.connections[("X", "H")].w
        if xh.shape[0] != actor_enc.n_inpt:
            raise ValueError(
                f"Input dim mismatch: encoder n_inpt={actor_enc.n_inpt}, "
                f"network expects {xh.shape[0]}"
            )

    load_actor_from_checkpoint(actor_net, ckpt_path)

    # monitors
    spikes = {}
    for layer in actor_net.layers:
        spikes[layer] = Monitor(actor_net.layers[layer], state_vars=["s"], time=timesteps)
        actor_net.add_monitor(spikes[layer], name=f"{layer}_spikes")

    # ------ env ------
    env = gym.make(
        "MountainCar-v0",
        render_mode=("human" if render else None),
        max_episode_steps=max_steps,
    )
    env.reset(seed=seed)

    lens, successes, final_positions, rewards = [], [], [], []

    for ep in tqdm(range(n_episodes), desc="TEST"):
        episode_reset(actor_net)
        state, _ = env.reset(seed=seed + ep)

        ep_reward = 0.0
        success_flag = False

        for ti in range(1, max_steps + 1):
            if isinstance(state, tuple):
                state = state[0]

            rates_X = actor_enc.rates(state)
            spk_X = poisson(datum=rates_X, time=sim_time, dt=dt)
            if spk_X.ndim == 2:
                spk_X = spk_X.unsqueeze(1)
            spk_X = spk_X.float()

            if actor_reset_activity:
                reset_activity_only(actor_net)

            actor_net.run(
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
        ckpt_path=CKPT_PATH,           # set to None to auto-pick best-ep-len from SAVE_DIR
        seed=0,
        n_episodes=100,
        max_steps=1000,
        time_sec=40.0,
        dt=0.5,

        encoder_num_points=100,
        encoder_rho=20000.0,
        encoder_pos_range=(-1.2, 0.6),
        encoder_vel_range=(-0.07, 0.07),

        n_neurons_hid=1000,
        inhib_hid=0.1,
        inhib_out=0.1,
        thresh_hid=-56.0,
        thresh_out=-63.0,
        refrac_hid=5,
        refrac_out=0,

        actor_reset_activity=True,
        epsilon=0.0,
        render=True,
    )
