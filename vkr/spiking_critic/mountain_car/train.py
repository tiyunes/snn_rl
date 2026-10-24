# ===========================
# MountainCar SNN Actor + Spiking Critic
# Actor:  pretrained hidden X->H (1000 neurons, InputLayer 200 inputs, time=40)
#         trainable H->Y (MSTDPET)
# Critic: SpikeCritic (MSTDPET)
#         Input: same InputLayer spikes as actor (shared encoder)
# Hidden layer: hidden_layers/mountain_car_1000n
# ===========================

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, REPO_ROOT)  # local modified bindsnet

import glob
import json
import random
import time as time_module
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import gymnasium as gym
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from bindsnet.encoding import poisson
from bindsnet.learning import NoOp, MSTDPET
from bindsnet.network import Network
from bindsnet.network.nodes import Input, DiehlAndCookNodesModified
from bindsnet.network.topology import Connection
from bindsnet.network.monitors import Monitor
from bindsnet.models import Diehl2LayersWTAOutput


# ---------------------------
# PATHS
# ---------------------------
DATA_PATH = os.path.join(REPO_ROOT, "data", "mountaincar_state_action.csv")

# Pretrained 1000-neuron hidden layer (InputLayer 200 inputs, time=40, thresh=-56, refrac=5)
HIDDEN_LAYER_PATH = os.path.join(REPO_ROOT, "hidden_layers", "mountain_car_1000n", "hidden_layer.pt")
SAVE_DIR = os.path.join(SCRIPT_DIR, "runs")


# ---------------------------
# HELPERS
# ---------------------------
def ensure_dirs(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)


def set_seeds(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def summarize_last_window(arr, window=200):
    if len(arr) == 0:
        return np.nan
    x = np.asarray(arr[-window:], dtype=np.float64)
    return float(np.mean(x))


def e_greedy_action(action_ind: int, epsilon=0.0, n_actions=3) -> int:
    probs = np.full(n_actions, epsilon / n_actions, dtype=np.float64)
    probs[action_ind] += (1.0 - epsilon)
    return int(np.random.choice(np.arange(n_actions), p=probs))


def discounted_step_reward_fixed_point(step_reward: float, gamma: float) -> float:
    """Bellman fixed point for constant per-step reward: V = r / (1 - gamma)."""
    if not (0.0 <= gamma < 1.0):
        raise ValueError(f"Expected gamma in [0, 1), got {gamma}")
    return float(step_reward / (1.0 - gamma))


# ---------------------------
# DATA / ENCODER
# ---------------------------
def load_states_from_csv(csv_path: str) -> np.ndarray:
    df = pd.read_csv(csv_path)
    required_cols = ["position", "velocity"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")
    return df[["position", "velocity"]].to_numpy(dtype=np.float32)


class InputLayerMountainCar:
    """
    Cartpole-style InputLayer encoder (no normalization).
    Matches the encoder used to pretrain the 1000-neuron hidden layer
    (hidden_layers/mountain_car_1000n/train_hidden.py).
      n_inpt = 2 * num_points
      rates  = rho * exp(-0.5 * (state - centers)^2 / h^2)  (no per-feature softmax)
    """

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
        self.n_inpt = self.num_features * self.num_points  # 200

    def encode_numpy(self, state):
        state = np.asarray(state, dtype=np.float32).reshape(self.num_features, 1)
        diff = (self.values - state) ** 2
        exp = np.exp(-0.5 * diff / (self.h ** 2))
        return (self.rho * exp.flatten()).astype(np.float32)

    def rates(self, state):
        return torch.tensor(self.encode_numpy(state), dtype=torch.float32, device=self.device)


def build_input_spike_train(encoder: InputLayerMountainCar, state, sim_time: float, dt: float) -> torch.Tensor:
    """Generate Poisson spike train. Returns [timesteps, 1, n_neurons]."""
    rates = encoder.rates(np.asarray(state, dtype=np.float32))
    spk = poisson(datum=rates, time=sim_time, dt=dt)
    if spk.ndim == 2:
        spk = spk.unsqueeze(1)
    return spk.float()


# ---------------------------
# RESET FUNCTIONS
# ---------------------------
def reset_activity_only(net):
    """Reset membrane/spike activity but preserve theta and MSTDPET traces."""
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
    """Episode-boundary reset: zeros activity + short-term MSTDPET traces,
    but PRESERVES theta so homeostasis accumulates across the training run."""
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
        # theta intentionally NOT zeroed

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
# STORE / RESTORE STATE
# ---------------------------
def _clone_tensor_if_exists(obj, attr):
    if hasattr(obj, attr):
        t = getattr(obj, attr)
        if torch.is_tensor(t):
            return t.clone().detach()
    return None


def store_state(net):
    state = {"layers": {}, "rules": {}}
    for lname, layer in net.layers.items():
        d = {}
        for attr in ["v", "s", "x", "refrac_count", "theta", "decay"]:
            t = _clone_tensor_if_exists(layer, attr)
            if t is not None:
                d[attr] = t
        state["layers"][lname] = d
    for ck, conn in net.connections.items():
        rule = conn.update_rule
        rd = {}
        for attr in ["p_plus", "p_minus", "eligibility", "eligibility_trace"]:
            t = _clone_tensor_if_exists(rule, attr)
            if t is not None:
                rd[attr] = t
        if rd:
            state["rules"][ck] = rd
    return state


def restore_state(net, state):
    for lname, d in state.get("layers", {}).items():
        if lname not in net.layers:
            continue
        layer = net.layers[lname]
        for attr, t in d.items():
            if hasattr(layer, attr) and torch.is_tensor(getattr(layer, attr)):
                getattr(layer, attr).copy_(t)
    for ck, rd in state.get("rules", {}).items():
        if ck not in net.connections:
            continue
        rule = net.connections[ck].update_rule
        for attr, t in rd.items():
            if hasattr(rule, attr) and torch.is_tensor(getattr(rule, attr)):
                getattr(rule, attr).copy_(t)


def get_spike_counts(spikes: dict, layer_name: str) -> torch.Tensor:
    s = spikes[layer_name].get("s")
    return s.sum(dim=0)[0].detach().clone()


# ---------------------------
# ACTOR
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

    # Freeze pretrained hidden layer
    if ("X", "H") in net.connections:
        net.connections[("X", "H")].update_rule = NoOp(
            connection=net.connections[("X", "H")],
            nu=net.connections[("X", "H")].nu,
        )

    conn_actor = net.connections[("H", "Y")]
    conn_actor.update_rule.a_minus = -1.0
    conn_actor.update_rule.a_plus = 1.0

    return net


# ---------------------------
# SPIKING CRITIC
# ---------------------------
class SpikeCritic(Network):
    """Single-population spiking value critic.

    V(s) = value_mean + value_std * gain * rate_p
    where rate_p = mean(Vp.x) / tc_trace_v  (normalized post-synaptic trace).

    Input layer 'XF' receives the same spikes as the actor.
    """

    def __init__(
        self,
        n_inpt: int,
        n_neurons_value: int = 1,
        dt: float = 0.5,
        critic_lr: float = 2e-6,
        thresh_v: float = -59.0,
        tc_trace_v: float = 30.0,
        tc_decay_v: float = 80.0,
        tc_trace_input: float = 20.0,
        tc_plus: float = 30.0,
        tc_minus: float = 30.0,
        tc_e_trace: float = 30.0,
        w_init: float = 0.2,
        wmax: float = 1.0,
        timesteps: int = 80,
        theta_plus: float = 0.0,
        tc_theta_decay: float = 10.0,
    ):
        super().__init__(dt=dt)

        XF = Input(n=n_inpt, traces=True, tc_trace=tc_trace_input)
        self.add_layer(XF, name="XF")

        Vp = DiehlAndCookNodesModified(
            n=n_neurons_value,
            traces=True,
            traces_additive=True,
            rest=-65.0,
            reset=-65.0,
            thresh=thresh_v,
            tc_decay=tc_decay_v,
            tc_trace=tc_trace_v,
            theta_plus=theta_plus,
            tc_theta_decay=tc_theta_decay,
            refrac=0,
            one_spike=False,
        )
        self.add_layer(Vp, name="Vp")

        conn_p = Connection(
            source=self.layers["XF"],
            target=self.layers["Vp"],
            w=w_init * torch.rand(n_inpt, n_neurons_value),
            update_rule=MSTDPET,
            nu=[0.0, critic_lr],
            tc_plus=tc_plus,
            tc_minus=tc_minus,
            tc_e_trace=tc_e_trace,
            wmin=0.0,
            wmax=wmax,
        )
        self.add_connection(conn_p, source="XF", target="Vp")

        self.connections[("XF", "Vp")].update_rule.a_minus = 0.0
        self.connections[("XF", "Vp")].update_rule.a_plus = 1.0

        self.add_monitor(
            Monitor(self.layers["Vp"], state_vars=["s"], time=timesteps),
            name="Vp_spikes",
        )


@torch.no_grad()
def read_value_components(critic_net):
    Vp = critic_net.layers["Vp"]
    tc_p = float(Vp.tc_trace) if not torch.is_tensor(Vp.tc_trace) else float(Vp.tc_trace.item())
    rate_p = float((Vp.x.mean(dim=1) / tc_p)[0].item())
    return rate_p


def signed_rate_to_value(rate_p, value_mean, value_std, gain):
    return float(value_mean + value_std * gain * rate_p)


@torch.no_grad()
def read_spike_stats(critic_net):
    sp = critic_net.monitors["Vp_spikes"].get("s")
    total = float(np.prod(sp.shape))
    frac = float(sp.float().sum().item() / (total + 1e-12))
    active = float((sp.sum(dim=0)[0] > 0).float().mean().item())
    return {"spike_frac_vp": frac, "active_neurons_vp": active}


def run_critic(
    critic_net,
    critic_spikes,
    sim_time,
    value_mean,
    value_std,
    gain,
    learning=False,
    reward_dict=None,
    reset_activity=False,
):
    if reset_activity:
        reset_activity_only(critic_net)

    critic_net.run(
        inputs={"XF": critic_spikes},
        time=sim_time,
        learning=learning,
        reward=reward_dict if reward_dict is not None else 0.0,
        step=0,
        one_step=False,
    )

    rate_p = read_value_components(critic_net)
    value_pred = signed_rate_to_value(rate_p, value_mean, value_std, gain)
    spike_stats = read_spike_stats(critic_net)

    return {"rate_p": rate_p, "value_pred": value_pred, **spike_stats}


# ---------------------------
# VALUE CALIBRATION
# ---------------------------
@torch.no_grad()
def calibrate_value_mean(
    critic_net,
    encoder,
    calibration_states: np.ndarray,
    sim_time: float,
    dt: float,
    value_std: float,
    gain: float,
    target_value: float,
    reset_activity: bool = False,
) -> float:
    """Shift value_mean so the initial critic output matches target_value on average."""
    rates = []
    for state in calibration_states:
        if reset_activity:
            reset_activity_only(critic_net)
        spk = build_input_spike_train(encoder, state, sim_time, dt)
        critic_net.run(
            inputs={"XF": spk},
            time=sim_time,
            learning=False,
            reward=0.0,
            step=0,
            one_step=False,
        )
        rate_p = read_value_components(critic_net)
        rates.append(float(rate_p))

    mean_rate = float(np.mean(rates))
    return float(target_value - value_std * gain * mean_rate)


# ---------------------------
# CHECKPOINT
# ---------------------------
def load_checkpoint(path: str, actor_net, critic_net) -> dict:
    """Load weights/theta from checkpoint into existing networks. Returns extra dict."""
    ckpt = torch.load(path, map_location="cpu")
    for key, conn in actor_net.connections.items():
        k = str(key)
        if k in ckpt.get("actor_connections", {}):
            conn.w.data.copy_(ckpt["actor_connections"][k]["w"])
    for lname, layer in actor_net.layers.items():
        if lname in ckpt.get("actor_layers", {}) and hasattr(layer, "theta"):
            layer.theta.data.copy_(ckpt["actor_layers"][lname]["theta"])
    for key, conn in critic_net.connections.items():
        k = str(key)
        if k in ckpt.get("critic_connections", {}):
            conn.w.data.copy_(ckpt["critic_connections"][k]["w"])
    for lname, layer in critic_net.layers.items():
        if lname in ckpt.get("critic_layers", {}) and hasattr(layer, "theta"):
            layer.theta.data.copy_(ckpt["critic_layers"][lname]["theta"])
    print(f"[CKPT] Loaded from: {path}  (saved at {ckpt.get('time', '?')})")
    return ckpt.get("extra", {})


def find_latest_checkpoint(save_dir: str) -> Optional[str]:
    """Возвращает путь к последнему периодическому ckpt_ep*.pt или None."""
    files = sorted(glob.glob(os.path.join(save_dir, "ckpt_ep?????.pt")))
    return files[-1] if files else None


def save_checkpoint(path: str, actor_net, critic_net, extra: dict):
    ckpt = {
        "time": time_module.strftime("%Y-%m-%d %H:%M:%S"),
        "actor_connections": {},
        "actor_layers": {},
        "critic_connections": {},
        "critic_layers": {},
        "extra": extra,
    }
    for key, conn in actor_net.connections.items():
        ckpt["actor_connections"][str(key)] = {"w": conn.w.detach().cpu()}
    for lname, layer in actor_net.layers.items():
        if hasattr(layer, "theta") and torch.is_tensor(layer.theta):
            ckpt["actor_layers"][lname] = {"theta": layer.theta.detach().cpu()}
    for key, conn in critic_net.connections.items():
        ckpt["critic_connections"][str(key)] = {"w": conn.w.detach().cpu()}
    for lname, layer in critic_net.layers.items():
        if hasattr(layer, "theta") and torch.is_tensor(layer.theta):
            ckpt["critic_layers"][lname] = {"theta": layer.theta.detach().cpu()}
    torch.save(ckpt, path)
    print(f"[CKPT] Saved to: {path}")


# ---------------------------
# MAIN
# ---------------------------
def main(
    seed=0,
    n_episodes=4000,
    max_steps=1000,
    time_sec=40.0,
    dt=0.5,
    gamma=0.99,
    epsilon=0.0,

    # actor (matches hidden-layer pretraining)
    actor_reset_activity=True,
    actor_lr_out=1e-4,
    actor_thresh_hid=-56.0,
    actor_thresh_out=-64.0,
    n_neurons_hid=1000,
    inhib_hid=0.1,
    inhib_out=0.1,
    refrac_hid=5,
    refrac_out=0,

    # encoder (matches the InputLayer used to train the hidden layer)
    encoder_num_points=100,
    encoder_rho=20000.0,
    encoder_pos_range=(-1.2, 0.6),
    encoder_vel_range=(-0.07, 0.07),

    # critic
    n_neurons_value=30,
    critic_lr=2e-3,
    thresh_v=-63.0,
    critic_w_init=0.2,
    gain=4.0,
    value_std=50.0,
    critic_tc_trace_v=30.0,
    critic_tc_decay_v=80.0,
    critic_tc_trace_input=20.0,
    critic_tc_plus=30.0,
    critic_tc_minus=30.0,
    critic_tc_e_trace=30.0,
    critic_theta_plus=0.0,
    critic_tc_theta_decay=10.0,
    critic_reset_activity=False,

    value_mean_mode="manual",
    value_mean=-100.0,
    calibration_num_states=128,
    nonterminal_reward=-1.0,
    terminal_bonus=0.0,
    shaping_scale=0.0,

    summary_every_episodes=25,
    save_every_episodes=25,
    render_mode=None,
    device=None,
    resume_from=None,
):
    ensure_dirs(SAVE_DIR)
    set_seeds(seed)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    print(f"[Device] {device}")
    print(f"[Hidden layer] {HIDDEN_LAYER_PATH}")
    if not os.path.exists(HIDDEN_LAYER_PATH):
        raise FileNotFoundError(
            f"HIDDEN_LAYER_PATH not found: {HIDDEN_LAYER_PATH}\n"
            f"Run hidden_layers/mountain_car_1000n/train_hidden.py first."
        )

    timesteps = int(round(time_sec / dt))
    sim_time = time_sec

    # ---- Env ----
    env = gym.make("MountainCar-v0", render_mode=render_mode, max_episode_steps=max_steps)
    env.reset(seed=seed)

    # ---- Encoder (cartpole-style InputLayer, 200 inputs) ----
    states_np = load_states_from_csv(DATA_PATH)

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

    # ---- Actor (1000 hidden neurons, pretrained X->H frozen) ----
    actor_net = build_actor_net(
        seed=seed,
        dt=dt,
        n_inpt=actor_enc.n_inpt,
        n_actions=3,
        n_neurons_hid=n_neurons_hid,
        inhib_hid=inhib_hid,
        inhib_out=inhib_out,
        lr_hid=1e-5,
        lr_out=actor_lr_out,
        w_init_hid=0.5,
        w_init_out=0.5,
        thresh_hid=actor_thresh_hid,
        thresh_out=actor_thresh_out,
        refrac_hid=refrac_hid,
        refrac_out=refrac_out,
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
    )
    actor_net.to(device)

    if ("X", "H") in actor_net.connections:
        xh = actor_net.connections[("X", "H")].w
        if xh.shape[0] != actor_enc.n_inpt:
            raise ValueError(
                f"Input dim mismatch: encoder n_inpt={actor_enc.n_inpt}, "
                f"hidden layer expects {xh.shape[0]}"
            )

    actor_spikes = {}
    for layer in actor_net.layers:
        actor_spikes[layer] = Monitor(
            actor_net.layers[layer], state_vars=["s"], time=timesteps
        )
        actor_net.add_monitor(actor_spikes[layer], name=f"{layer}_spikes")

    print(f"[Actor] hidden={actor_net.n_neurons_hid}, output={actor_net.n_neurons_out}")

    # ---- Spiking Critic ----
    critic_net = SpikeCritic(
        n_inpt=actor_enc.n_inpt,
        n_neurons_value=n_neurons_value,
        dt=dt,
        critic_lr=critic_lr,
        thresh_v=thresh_v,
        tc_trace_v=critic_tc_trace_v,
        tc_decay_v=critic_tc_decay_v,
        tc_trace_input=critic_tc_trace_input,
        tc_plus=critic_tc_plus,
        tc_minus=critic_tc_minus,
        tc_e_trace=critic_tc_e_trace,
        w_init=critic_w_init,
        wmax=1.0,
        timesteps=timesteps,
        theta_plus=critic_theta_plus,
        tc_theta_decay=critic_tc_theta_decay,
    )
    print(f"[Critic] n_neurons_value={n_neurons_value}")

    # ---- Resume from checkpoint ----
    resume_extra = {}
    if resume_from is not None:
        ckpt_path = (
            find_latest_checkpoint(SAVE_DIR) if resume_from == "auto" else resume_from
        )
        if ckpt_path is None:
            print("[RESUME] Чекпойнт не найден, стартуем с нуля.")
        else:
            resume_extra = load_checkpoint(ckpt_path, actor_net, critic_net)

    # ---- Calibrate value_mean ----
    if resume_from is not None and "value_mean" in resume_extra:
        value_mean = float(resume_extra["value_mean"])
        print(f"[RESUME] value_mean restored from checkpoint: {value_mean:.4f}")
    elif value_mean_mode == "bellman_centered":
        target_value = discounted_step_reward_fixed_point(
            step_reward=nonterminal_reward, gamma=gamma
        )
        idx = np.random.choice(
            len(states_np),
            size=min(int(calibration_num_states), len(states_np)),
            replace=False,
        )
        calib_states = states_np[idx]
        value_mean = calibrate_value_mean(
            critic_net=critic_net,
            encoder=actor_enc,
            calibration_states=calib_states,
            sim_time=sim_time,
            dt=dt,
            value_std=value_std,
            gain=gain,
            target_value=target_value,
            reset_activity=critic_reset_activity,
        )
        print(
            f"[CALIB] value_mean={value_mean:.4f} for "
            f"Bellman target={target_value:.2f} (r={nonterminal_reward}, γ={gamma})"
        )
    elif value_mean_mode == "manual":
        if value_mean is None:
            raise ValueError("Provide value_mean when value_mean_mode='manual'")
        value_mean = float(value_mean)
    else:
        raise ValueError(f"Unknown value_mean_mode={value_mean_mode!r}")

    # ---- Logs ----
    rewards_arr = []
    ep_len_arr = []
    success_arr = []
    final_positions_arr = []
    values_arr = []
    td_errors_arr = []
    spike_frac_arr = []
    active_neurons_arr = []

    running_reward = float(discounted_step_reward_fixed_point(nonterminal_reward, gamma))
    best_mean_ep_len = np.inf
    best_ep_len = np.inf
    start_ep = 1

    if resume_extra:
        ep_len_arr = list(resume_extra.get("all_ep_lens", []))
        rewards_arr = list(resume_extra.get("all_rewards", []))
        success_arr = list(resume_extra.get("all_success", []))
        final_positions_arr = list(resume_extra.get("all_final_positions", []))
        start_ep = int(resume_extra.get("episode", 0)) + 1
        running_reward = float(resume_extra.get("running_reward", running_reward))
        if ep_len_arr:
            best_ep_len = float(min(ep_len_arr))
        best_mean_ep_len = float(resume_extra.get("mean_ep_len_100", np.inf))
        print(f"[RESUME] Continuing from episode {start_ep}, best_ep_len={best_ep_len}, best_mean={best_mean_ep_len:.2f}")

    # ---- Training loop ----
    for ep in tqdm(range(start_ep, start_ep + n_episodes), desc="Train"):
        episode_reset(actor_net)
        episode_reset(critic_net)
        state, _ = env.reset(seed=seed + ep)

        ep_reward = 0.0
        success_flag = False

        for ti in range(1, max_steps + 1):
            if isinstance(state, tuple):
                state = state[0]

            # ---- Spikes for s_t (shared for actor and critic) ----
            spk_X = build_input_spike_train(actor_enc, state=state, sim_time=sim_time, dt=dt)
            actor_inpts_t = {"X": spk_X}

            # ---- A-pass actor: choose action ----
            actor_snap0 = store_state(actor_net)
            if actor_reset_activity:
                reset_activity_only(actor_net)
            actor_net.run(
                inputs=actor_inpts_t,
                time=sim_time,
                learning=False,
                reward=0.0,
                step=ti - 1,
                one_step=False,
            )
            y_counts = get_spike_counts(actor_spikes, "Y")
            action_pred = int(torch.argmax(y_counts).item())
            action = e_greedy_action(action_pred, epsilon=epsilon, n_actions=3)

            # ---- Critic eval on s_t ----
            critic_snap0 = store_state(critic_net)
            out_s = run_critic(
                critic_net=critic_net,
                critic_spikes=spk_X,
                sim_time=sim_time,
                value_mean=value_mean,
                value_std=value_std,
                gain=gain,
                learning=False,
                reward_dict=None,
                reset_activity=critic_reset_activity,
            )
            V_s = float(out_s["value_pred"])

            # ---- Env step ----
            next_state, _, terminated, _, _ = env.step(int(action))

            if terminated:
                reward_used = float(terminal_bonus)
                success_flag = True
            else:
                reward_used = float(nonterminal_reward)
                if shaping_scale != 0.0:
                    cur_pos = float(state[0])
                    next_pos = float(next_state[0])
                    reward_used += float(shaping_scale * (gamma * next_pos - cur_pos))

            # ---- Critic eval on s_{t+1} (from same baseline critic_snap0) ----
            restore_state(critic_net, critic_snap0)
            spk_X_next = build_input_spike_train(actor_enc, state=next_state, sim_time=sim_time, dt=dt)
            out_ns = run_critic(
                critic_net=critic_net,
                critic_spikes=spk_X_next,
                sim_time=sim_time,
                value_mean=value_mean,
                value_std=value_std,
                gain=gain,
                learning=False,
                reward_dict=None,
                reset_activity=critic_reset_activity,
            )
            V_ns = float(out_ns["value_pred"])

            # ---- TD error ----
            if terminated:
                target = reward_used
            else:
                target = reward_used + gamma * V_ns
            delta = float(target - V_s)

            td_mod_signal = float(np.clip(delta / (value_std + 1e-8), -3.0, 3.0))

            # ---- Actor learning-pass on s_t ----
            restore_state(actor_net, actor_snap0)
            if actor_reset_activity:
                reset_activity_only(actor_net)
            actor_net.run(
                inputs=actor_inpts_t,
                time=sim_time,
                learning=True,
                reward=td_mod_signal,
                step=ti - 1,
                one_step=False,
            )

            # ---- Critic learning-pass on s_t ----
            restore_state(critic_net, critic_snap0)
            reward_dict = {
                critic_net.connections[("XF", "Vp")]: td_mod_signal,
            }
            _ = run_critic(
                critic_net=critic_net,
                critic_spikes=spk_X,
                sim_time=sim_time,
                value_mean=value_mean,
                value_std=value_std,
                gain=gain,
                learning=True,
                reward_dict=reward_dict,
                reset_activity=critic_reset_activity,
            )

            # ---- Logs ----
            ep_reward += reward_used
            values_arr.append(float(V_s))
            td_errors_arr.append(float(delta))
            spike_frac_arr.append(float(out_s["spike_frac_vp"]))
            active_neurons_arr.append(float(out_s["active_neurons_vp"]))

            state = next_state

            if terminated:
                break

        running_reward = 0.05 * ep_reward + (1.0 - 0.05) * running_reward

        rewards_arr.append(ep_reward)
        ep_len_arr.append(ti)
        success_arr.append(int(success_flag))
        final_positions_arr.append(float(state[0]))

        mean_ep_len_100 = summarize_last_window(ep_len_arr, 100)

        def _ckpt_extra():
            return {
                "episode": ep,
                "running_reward": float(running_reward),
                "all_ep_lens": ep_len_arr,
                "all_rewards": rewards_arr,
                "all_success": success_arr,
                "all_final_positions": final_positions_arr,
                "mean_ep_len_100": float(mean_ep_len_100),
                "value_mean": float(value_mean),
                "value_std": float(value_std),
                "gain": float(gain),
                "n_neurons_value": int(n_neurons_value),
                "critic_lr": float(critic_lr),
                "actor_lr_out": float(actor_lr_out),
                "gamma": float(gamma),
                "epsilon": float(epsilon),
                "encoder_num_points": int(encoder_num_points),
                "encoder_rho": float(encoder_rho),
                "encoder_pos_range": list(encoder_pos_range),
                "encoder_vel_range": list(encoder_vel_range),
                "hidden_layer_path": HIDDEN_LAYER_PATH,
            }

        if ti < best_ep_len:
            best_ep_len = ti
            save_checkpoint(
                path=os.path.join(SAVE_DIR, f"ckpt_ep{ep:05d}_best_ep_len_{ti}.pt"),
                actor_net=actor_net,
                critic_net=critic_net,
                extra=_ckpt_extra(),
            )
            print(f"[BEST EP LEN] ep={ep} ep_len={best_ep_len}")

        if ep % summary_every_episodes == 0:
            print(
                f"EP={ep} | len={ti} | ep_reward={ep_reward:.2f} | "
                f"mean len(200)={summarize_last_window(ep_len_arr, 200):.2f} | "
                f"mean reward(200)={summarize_last_window(rewards_arr, 200):.2f} | "
                f"success(200)={summarize_last_window(success_arr, 200):.3f} | "
                f"mean pos(200)={summarize_last_window(final_positions_arr, 200):.4f} | "
                f"mean V(200)={summarize_last_window(values_arr, 200):.4f} | "
                f"mean delta(200)={summarize_last_window(td_errors_arr, 200):.4f} | "
                f"sfrac(200)={summarize_last_window(spike_frac_arr, 200):.4f}"
            )

        if np.isfinite(mean_ep_len_100) and mean_ep_len_100 < best_mean_ep_len:
            best_mean_ep_len = mean_ep_len_100
            save_checkpoint(
                path=os.path.join(SAVE_DIR, "ckpt_best.pt"),
                actor_net=actor_net,
                critic_net=critic_net,
                extra=_ckpt_extra(),
            )
            print(f"[BEST] ep={ep} mean_ep_len(100)={best_mean_ep_len:.2f}")

        if ep % save_every_episodes == 0:
            save_checkpoint(
                path=os.path.join(SAVE_DIR, f"ckpt_ep{ep:05d}.pt"),
                actor_net=actor_net,
                critic_net=critic_net,
                extra=_ckpt_extra(),
            )

    env.close()

    # ---- Final plots ----
    plt.figure(figsize=(8, 4))
    plt.plot(ep_len_arr)
    plt.title("Episode length")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.plot(rewards_arr)
    plt.title("Episode reward")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.plot(pd.Series(success_arr).rolling(100, min_periods=1).mean())
    plt.title("Success rate rolling-100")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.plot(final_positions_arr)
    plt.title("Final position per episode")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    if values_arr:
        plt.figure(figsize=(8, 4))
        plt.plot(values_arr[-2000:])
        plt.title("V (spiking critic) last 2000 steps")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    if td_errors_arr:
        plt.figure(figsize=(8, 4))
        plt.plot(td_errors_arr[-2000:])
        plt.title("TD error last 2000 steps")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    if spike_frac_arr:
        plt.figure(figsize=(8, 4))
        plt.plot(pd.Series(spike_frac_arr).rolling(500, min_periods=1).mean())
        plt.title("Critic Vp spike fraction (rolling-500)")
        plt.grid(True)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main(
        seed=0,
        n_episodes=4000,
        max_steps=1000,
        time_sec=40.0,
        dt=0.5,
        gamma=0.99,
        epsilon=0.0,

        # actor (matches hidden-layer pretraining)
        actor_reset_activity=True,
        actor_lr_out=5e-3,
        actor_thresh_hid=-56.0,
        actor_thresh_out=-63.0,
        n_neurons_hid=1000,
        inhib_hid=0.1,
        inhib_out=0.1,
        refrac_hid=5,
        refrac_out=0,

        # encoder
        encoder_num_points=100,
        encoder_rho=20000.0,
        encoder_pos_range=(-1.2, 0.6),
        encoder_vel_range=(-0.07, 0.07),

        # critic
        n_neurons_value=30,
        critic_lr=3e-5,
        thresh_v=-64.5,
        critic_w_init=0.2,
        gain=4.0,
        value_std=50.0,
        critic_tc_trace_v=30.0,
        critic_tc_decay_v=80.0,
        critic_tc_trace_input=20.0,
        critic_tc_plus=30.0,
        critic_tc_minus=30.0,
        critic_tc_e_trace=30.0,
        critic_theta_plus=0.0,
        critic_tc_theta_decay=10.0,
        critic_reset_activity=False,

        value_mean_mode="manual",
        value_mean=-100.0,
        calibration_num_states=128,
        nonterminal_reward=-1.0,
        terminal_bonus=0.0,
        shaping_scale=0.0,

        summary_every_episodes=25,
        save_every_episodes=25,
        render_mode="human",
        device=None,
        # None  — старт с нуля
        # "auto" — последний ckpt_ep?????.pt из SAVE_DIR
        # путь   — конкретный .pt файл
        resume_from=None,
    )
