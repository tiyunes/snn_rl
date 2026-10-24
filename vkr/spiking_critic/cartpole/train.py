# ===========================
# CartPole online actor-critic
# Spiking actor (Diehl2LayersWTAOutput, 60-input GRF, 500-neuron pretrained
# hidden layer from hidden_layers/cartpole_500n) +
# single-population spiking critic (DiehlAndCookNodesModified + MSTDPET).
#
# read_value_components reads V from the population spike rate over the
# whole simulation window (Vp_spikes monitor).
# ===========================

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, REPO_ROOT)  # local modified bindsnet

import json
import time
import random
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional

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
HIDDEN_LAYER_PATH = os.path.join(REPO_ROOT, "hidden_layers", "cartpole_500n", "hidden_layer.pt")
SAVE_DIR = os.path.join(SCRIPT_DIR, "runs")
Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)


# ---------------------------
# HELPERS
# ---------------------------
def set_seeds(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def summarize_last_window(arr, window=200):
    if len(arr) == 0:
        return np.nan
    x = np.asarray(arr[-window:], dtype=np.float64)
    return float(np.mean(x))


def sanitize_run_name(name: Optional[str]) -> Optional[str]:
    if name is None:
        return None
    keep = []
    for ch in str(name):
        if ch.isalnum() or ch in ("-", "_", "."):
            keep.append(ch)
        else:
            keep.append("_")
    sanitized = "".join(keep).strip("_.")
    return sanitized or "run"


def make_run_dir(base_dir: str, run_name: Optional[str] = None) -> str:
    path = Path(base_dir)
    if run_name is not None:
        path = path / sanitize_run_name(run_name)
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def theta_max_in_net(net) -> float:
    theta_max = 0.0
    for layer in net.layers.values():
        if hasattr(layer, "theta") and torch.is_tensor(layer.theta):
            theta_max = max(theta_max, float(layer.theta.max().item()))
    return theta_max


def theta_stats_in_net(net) -> Dict[str, float]:
    theta_values = []
    for layer in net.layers.values():
        if hasattr(layer, "theta") and torch.is_tensor(layer.theta):
            theta_values.append(layer.theta.detach().cpu().reshape(-1).numpy())
    if len(theta_values) == 0:
        return {"theta_mean": np.nan, "theta_std": np.nan, "theta_min": np.nan, "theta_max": np.nan}
    theta_np = np.concatenate(theta_values, axis=0).astype(np.float64)
    return {
        "theta_mean": float(theta_np.mean()),
        "theta_std": float(theta_np.std()),
        "theta_min": float(theta_np.min()),
        "theta_max": float(theta_np.max()),
    }


def discounted_step_reward_fixed_point(step_reward: float, gamma: float) -> float:
    """V = r / (1 - gamma) for a constant per-step reward.

    Used to Bellman-center the untrained critic so TD errors are not
    systematically biased at initialization.
    """
    if not (0.0 <= gamma < 1.0):
        raise ValueError(f"Expected gamma in [0, 1), got {gamma}")
    return float(step_reward / (1.0 - gamma))


def reset_activity_only(net):
    """Reset v / s / x / refrac_count, preserve theta and learning traces.
    Used between within-step critic passes (3-pass TD evaluation)."""
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
    """Episode-boundary reset: zero membrane / spike activity AND short-term
    MSTDPET traces (p_plus, p_minus, eligibility, eligibility_trace), but
    PRESERVE theta so homeostasis accumulates across the run.
    """
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
        if len(rd) > 0:
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


def e_greedy_action(action_ind: int, n_actions: int = 2, epsilon: float = 0.1) -> int:
    prob = (epsilon / float(n_actions)) * np.ones(n_actions)
    prob[action_ind] = (1.0 - epsilon) + (epsilon / float(n_actions))
    return int(np.random.choice(n_actions, 1, p=prob))


def get_scale(mean_reward: float, ep_reward: float = 0.0, max_steps: int = 200) -> float:
    reward = max(mean_reward, ep_reward)
    if reward >= max_steps:
        return 0.0
    if reward <= 0.0:
        return 1.0
    return 1.0 / reward


def recent_reward_lr_multiplier(
    recent_rewards,
    window: int = 10,
    saturate_at: float = 180.0,
    min_scale: float = 0.001,
    power: float = 2.0,
) -> float:
    """Multiplier in [min_scale, 1.0] derived from mean of last `window`
    completed episode rewards. Applied to BOTH actor and critic LRs so a
    few high-reward episodes don't get clobbered by ongoing learning noise.

    mean=0           -> 1.0     (full LR)
    mean=saturate_at -> 0       (clamped to min_scale)
    Reacts within `window` episodes (fast vs the 0.05/0.95 running_reward EMA).
    """
    if len(recent_rewards) == 0:
        return 1.0
    w = min(int(window), len(recent_rewards))
    mean_r = float(np.mean(recent_rewards[-w:]))
    norm = max(0.0, 1.0 - mean_r / float(saturate_at))
    return float(max(min_scale, norm ** power))


def save_checkpoint(path: str, actor_net, critic_net, extra: dict):
    ckpt = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "actor_connections": {},
        "actor_layers": {},
        "critic_connections": {},
        "critic_layers": {},
        "extra": extra,
    }
    for key, conn in actor_net.connections.items():
        ckpt["actor_connections"][str(key)] = {
            "w": conn.w.detach().cpu(),
            "nu": getattr(conn, "nu", None),
        }
    for lname, layer in actor_net.layers.items():
        layer_pack = {}
        if hasattr(layer, "theta") and torch.is_tensor(layer.theta):
            layer_pack["theta"] = layer.theta.detach().cpu()
        if len(layer_pack) > 0:
            ckpt["actor_layers"][lname] = layer_pack
    for key, conn in critic_net.connections.items():
        ckpt["critic_connections"][str(key)] = {
            "w": conn.w.detach().cpu(),
            "nu": getattr(conn, "nu", None),
        }
    for lname, layer in critic_net.layers.items():
        layer_pack = {}
        if hasattr(layer, "theta") and torch.is_tensor(layer.theta):
            layer_pack["theta"] = layer.theta.detach().cpu()
        if len(layer_pack) > 0:
            ckpt["critic_layers"][lname] = layer_pack
    torch.save(ckpt, path)
    print(f"[CKPT] Saved to: {path}")


def write_checkpoint_log(path: str, records: List[Dict[str, object]]):
    pd.DataFrame(records).to_csv(path, index=False)


# ---------------------------
# CARTPOLE INPUT ENCODING
# Gaussian receptive field grid:
# 4 features × 15 centers = 60 input neurons.
# ---------------------------
class InputLayer:
    def __init__(self, rho: float = 5000.0, num_points: int = 15, device: str = "cpu"):
        self.num_features = 4
        self.rho = float(rho)
        self.num_points = int(num_points)
        self.device = device

        self.h = np.zeros(self.num_features)

        x_values = np.linspace(-2.5, 2.5, self.num_points)
        self.h[0] = x_values[1] - x_values[0]

        vel_values = np.linspace(-1.6, 1.6, self.num_points)
        self.h[1] = vel_values[1] - vel_values[0]

        theta_values = np.linspace(-0.45, 0.45, self.num_points)
        self.h[2] = theta_values[1] - theta_values[0]

        theta_vel_values = np.linspace(-2.6, 2.6, self.num_points)
        self.h[3] = theta_vel_values[1] - theta_vel_values[0]

        self.h = self.h.reshape(self.num_features, 1)
        self.values = np.vstack((x_values, vel_values, theta_values, theta_vel_values))

        self.n_inpt = self.num_features * self.num_points

    def encode_numpy(self, state) -> np.ndarray:
        state = np.asarray(state, dtype=np.float32).reshape(self.num_features, 1)
        diff = (self.values - state) ** 2
        exp = np.exp(-0.5 * diff / (self.h ** 2))
        return (self.rho * exp.flatten()).astype(np.float32)

    def rates(self, state) -> torch.Tensor:
        return torch.tensor(self.encode_numpy(state), dtype=torch.float32, device=self.device)


def build_grf_spike_train(encoder: InputLayer, state, sim_time: float, dt: float) -> torch.Tensor:
    rates = encoder.rates(np.asarray(state, dtype=np.float32))
    spk = poisson(datum=rates, time=sim_time, dt=dt)
    if spk.ndim == 2:
        spk = spk.unsqueeze(1)
    return spk.float()


# ---------------------------
# ACTOR (CartPole)
# ---------------------------
def build_actor_net(
    seed: int = 0,
    dt: float = 0.5,
    n_inpt: int = 60,
    n_actions: int = 2,
    n_neurons_hid: int = 500,
    inhib_hid: float = 0.1,
    inhib_out: float = 0.1,
    lr_hid: float = 1e-5,
    lr_out: float = 1e-3,
    w_init_hid: float = 0.5,
    w_init_out: float = 0.5,
    thresh_hid: float = -58.0,
    thresh_out: float = -64.0,
    refrac_hid: int = 5,
    refrac_out: int = 0,
    one_spike_hid: bool = False,
    one_spike_out: bool = True,
    tc_trace_input: float = 20.0,
    tc_decay_hid: float = 15.0,
    tc_decay_out: float = 15.0,
    tc_trace_hid: float = 10.0,
    tc_trace_out: float = 10.0,
    tc_plus_out: float = 10.0,
    tc_minus_out: float = 10.0,
    tc_e_trace_out: float = 15.0,
    hidden_layer_path: str = HIDDEN_LAYER_PATH,
):
    set_seeds(seed)

    net = Diehl2LayersWTAOutput(
        n_inpt=n_inpt,
        n_neurons_hid=n_neurons_hid,
        n_neurons_out=n_actions,
        inh_hid=inhib_hid,
        inh_out=inhib_out,
        dt=dt,
        nu_hid=[0.25 * lr_hid, lr_hid],
        nu_out=[0.4 * lr_out, lr_out],
        reduction=None,
        wmin=0.0,
        wmax=1.0,
        norm=None,
        theta_plus=0.0,
        tc_theta_decay=1e7,
        inpt_shape=None,
        hidden_layer_path=hidden_layer_path,
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
    )

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
# SPIKING CRITIC (single population)
# V(s) = value_mean + value_std * gain * rate_p
# where rate_p = mean(post-synaptic trace) / tc_trace.
# ---------------------------
class SpikeCritic(Network):
    def __init__(
        self,
        n_inpt: int,
        n_neurons_value: int = 15,
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
        timesteps: int = 40,
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

        self.add_monitor(Monitor(self.layers["Vp"], state_vars=["s"], time=timesteps), name="Vp_spikes")


@torch.no_grad()
def read_value_components(net, sim_time):
    # Population mean firing rate over the whole simulation window:
    # avoids the high-variance single-sample read of Vp.x at t=sim_time.
    spikes = net.monitors["Vp_spikes"].get("s")  # [T, batch, n_neurons]
    n_neurons = spikes.shape[-1]
    rate_p = float(spikes.float().sum().item()) / (float(n_neurons) * float(sim_time))
    return rate_p, 0.0, rate_p


def signed_rate_to_value(signed_rate, value_mean, value_std, gain):
    return float(value_mean + value_std * gain * signed_rate)


@torch.no_grad()
def read_spike_stats(net):
    sp_vp = net.monitors["Vp_spikes"].get("s")
    total_steps_vp = float(np.prod(sp_vp.shape))
    frac_vp = float(sp_vp.float().sum().item() / total_steps_vp)
    active_neurons_vp = float((sp_vp.sum(dim=0)[0] > 0).float().mean().item())
    return {
        "spike_frac_vp": frac_vp,
        "active_neurons_vp": active_neurons_vp,
    }


def run_actor_pass(actor_net, actor_inputs, sim_time, step, learning=False, reward=0.0, reset_activity=False):
    if reset_activity:
        reset_activity_only(actor_net)
    actor_net.run(
        inputs=actor_inputs,
        time=sim_time,
        learning=learning,
        reward=reward,
        step=step,
        one_step=False,
    )


def run_critic_state(
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
    rate_p, rate_n, signed_rate = read_value_components(critic_net, sim_time)
    value_pred = signed_rate_to_value(signed_rate, value_mean, value_std, gain)
    spike_stats = read_spike_stats(critic_net)
    return {
        "rate_p": rate_p,
        "rate_n": rate_n,
        "signed_rate": signed_rate,
        "value_pred": value_pred,
        **spike_stats,
    }


# ---------------------------
# CALIBRATION
# Sample states uniformly within the encoder grid range to seed the
# untrained critic mean around the Bellman fixed point r/(1-gamma).
# ---------------------------
def sample_calibration_states(encoder: InputLayer, num_states: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rows = []
    for f in range(encoder.num_features):
        low = float(encoder.values[f, 0])
        high = float(encoder.values[f, -1])
        rows.append(rng.uniform(low, high, size=num_states).astype(np.float32))
    return np.stack(rows, axis=1)


@torch.no_grad()
def estimate_mean_signed_rate(critic_net, state_to_spikes, states, sim_time, reset_activity=True):
    rates = []
    for s in states:
        if reset_activity:
            reset_activity_only(critic_net)
        spk = state_to_spikes(np.asarray(s, dtype=np.float32))
        critic_net.run(
            inputs={"XF": spk},
            time=sim_time,
            learning=False,
            reward=0.0,
            step=0,
            one_step=False,
        )
        _, _, signed_rate = read_value_components(critic_net, sim_time)
        rates.append(float(signed_rate))
    if len(rates) == 0:
        raise ValueError("Need >= 1 calibration state")
    return float(np.mean(rates))


@torch.no_grad()
def calibrate_value_mean(critic_net, state_to_spikes, calibration_states, sim_time, value_std, gain, target_value, reset_activity=True):
    mean_rate = estimate_mean_signed_rate(
        critic_net=critic_net,
        state_to_spikes=state_to_spikes,
        states=calibration_states,
        sim_time=sim_time,
        reset_activity=reset_activity,
    )
    return float(target_value - value_std * gain * mean_rate)


@torch.no_grad()
def evaluate_critic_validation_set(critic_net, state_to_spikes, validation_states, sim_time, value_mean, value_std, gain, reset_activity=True):
    if validation_states is None or len(validation_states) == 0:
        raise ValueError("validation_states must be non-empty")

    snapshot = store_state(critic_net)

    values, signed_rates = [], []
    spike_frac_vp, active_vp = [], []

    for s in validation_states:
        restore_state(critic_net, snapshot)
        if reset_activity:
            reset_activity_only(critic_net)
        spk = state_to_spikes(np.asarray(s, dtype=np.float32))
        critic_net.run(
            inputs={"XF": spk},
            time=sim_time,
            learning=False,
            reward=0.0,
            step=0,
            one_step=False,
        )
        rate_p, _, signed_rate = read_value_components(critic_net, sim_time)
        stats = read_spike_stats(critic_net)
        values.append(signed_rate_to_value(signed_rate, value_mean, value_std, gain))
        signed_rates.append(float(signed_rate))
        spike_frac_vp.append(float(stats["spike_frac_vp"]))
        active_vp.append(float(stats["active_neurons_vp"]))

    restore_state(critic_net, snapshot)

    v = np.asarray(values, dtype=np.float64)
    sr = np.asarray(signed_rates, dtype=np.float64)
    sf = np.asarray(spike_frac_vp, dtype=np.float64)
    an = np.asarray(active_vp, dtype=np.float64)
    th = theta_stats_in_net(critic_net)

    return {
        "critic_eval_num_states": int(len(validation_states)),
        "value_mean_eval": float(v.mean()),
        "value_std_eval": float(v.std()),
        "value_min_eval": float(v.min()),
        "value_max_eval": float(v.max()),
        "value_range_eval": float(v.max() - v.min()),
        "signed_rate_mean_eval": float(sr.mean()),
        "signed_rate_std_eval": float(sr.std()),
        "vp_spike_frac_mean_eval": float(sf.mean()),
        "vp_active_mean_eval": float(an.mean()),
        "theta_mean_critic": th["theta_mean"],
        "theta_std_critic": th["theta_std"],
        "theta_min_critic": th["theta_min"],
        "theta_max_critic": th["theta_max"],
    }


# ---------------------------
# MAIN
# ---------------------------
def main(
    seed: int = 0,
    n_episodes: int = 7000,
    max_steps: int = 200,
    time_sec: float = 20.0,
    dt: float = 0.5,
    gamma: float = 0.99,
    epsilon: float = 0.1,

    # actor
    rho: float = 5000.0,
    n_neurons_hid: int = 500,
    inhib_hid: float = 0.1,
    inhib_out: float = 0.1,
    actor_lr_hid: float = 1e-5,
    actor_lr_out: float = 1e-3,
    actor_w_init_hid: float = 0.5,
    actor_w_init_out: float = 0.5,
    actor_thresh_hid: float = -58.0,
    actor_thresh_out: float = -64.0,
    actor_refrac_hid: int = 5,
    actor_refrac_out: int = 0,
    actor_one_spike_hid: bool = False,
    actor_one_spike_out: bool = True,
    actor_tc_trace_input: float = 20.0,
    actor_tc_decay_hid: float = 15.0,
    actor_tc_decay_out: float = 15.0,
    actor_tc_trace_hid: float = 10.0,
    actor_tc_trace_out: float = 10.0,
    actor_tc_plus_out: float = 10.0,
    actor_tc_minus_out: float = 10.0,
    actor_tc_e_trace_out: float = 15.0,
    actor_reset_activity: bool = True,
    hidden_layer_path: str = HIDDEN_LAYER_PATH,

    # critic
    n_neurons_value: int = 15,
    critic_lr: float = 2e-6,
    thresh_v: float = -59.0,
    critic_w_init: float = 0.2,
    critic_wmax: float = 1.0,
    gain: float = 4.0,
    critic_tc_trace_v: float = 30.0,
    critic_tc_decay_v: float = 80.0,
    critic_tc_trace_input: float = 20.0,
    critic_tc_plus: float = 30.0,
    critic_tc_minus: float = 30.0,
    critic_tc_e_trace: float = 30.0,
    critic_theta_plus: float = 0.0,
    critic_tc_theta_decay: float = 10.0,
    critic_reset_activity: bool = False,

    # value readout
    value_mean: Optional[float] = None,
    value_std: float = 50.0,
    value_mean_mode: str = "bellman_centered",
    calibration_num_states: int = 128,

    # rewards
    nonterminal_reward: float = 1.0,
    terminal_reward: float = 0.0,
    use_env_reward: bool = True,

    # adaptive LR throttling based on recent episode rewards
    # (applied as multiplier to BOTH actor and critic LRs)
    lr_recent_window: int = 10,
    lr_recent_saturate_at: float = 180.0,
    lr_recent_min_scale: float = 0.01,
    lr_recent_power: float = 2.0,

    # logging
    summary_every_episodes: int = 25,
    save_every_episodes: int = 100,
    critic_eval_every_episodes: int = 100,
    critic_eval_num_states: int = 128,
    critic_eval_seed_offset: int = 1000,
    save_dir: str = SAVE_DIR,
    run_name: Optional[str] = None,
    plot_results: bool = True,
):
    set_seeds(seed)

    run_dir = make_run_dir(save_dir, run_name)
    timesteps = int(round(time_sec / dt))
    sim_time = time_sec

    actor_time_constants = {
        "actor_tc_trace_input": actor_tc_trace_input,
        "actor_tc_decay_hid": actor_tc_decay_hid,
        "actor_tc_decay_out": actor_tc_decay_out,
        "actor_tc_trace_hid": actor_tc_trace_hid,
        "actor_tc_trace_out": actor_tc_trace_out,
        "actor_tc_plus_out": actor_tc_plus_out,
        "actor_tc_minus_out": actor_tc_minus_out,
        "actor_tc_e_trace_out": actor_tc_e_trace_out,
    }
    critic_time_constants = {
        "critic_tc_trace_v": critic_tc_trace_v,
        "critic_tc_decay_v": critic_tc_decay_v,
        "critic_tc_trace_input": critic_tc_trace_input,
        "critic_tc_plus": critic_tc_plus,
        "critic_tc_minus": critic_tc_minus,
        "critic_tc_e_trace": critic_tc_e_trace,
        "critic_theta_plus": critic_theta_plus,
        "critic_tc_theta_decay": critic_tc_theta_decay,
    }

    run_config = {
        "run_name": run_name,
        "run_dir": run_dir,
        "seed": seed,
        "n_episodes": n_episodes,
        "max_steps": max_steps,
        "time_sec": time_sec,
        "dt": dt,
        "timesteps": timesteps,
        "gamma": gamma,
        "epsilon": epsilon,
        "rho": rho,
        "n_neurons_hid": n_neurons_hid,
        "actor_lr_hid": actor_lr_hid,
        "actor_lr_out": actor_lr_out,
        "actor_reset_activity": actor_reset_activity,
        "actor_time_constants": actor_time_constants,
        "n_neurons_value": n_neurons_value,
        "critic_lr": critic_lr,
        "thresh_v": thresh_v,
        "critic_w_init": critic_w_init,
        "critic_wmax": critic_wmax,
        "gain": gain,
        "value_std": value_std,
        "value_mean_mode": value_mean_mode,
        "requested_value_mean": value_mean,
        "calibration_num_states": calibration_num_states,
        "nonterminal_reward": nonterminal_reward,
        "terminal_reward": terminal_reward,
        "use_env_reward": use_env_reward,
        "critic_reset_activity": critic_reset_activity,
        "critic_time_constants": critic_time_constants,
        "summary_every_episodes": summary_every_episodes,
        "save_every_episodes": save_every_episodes,
        "critic_eval_every_episodes": critic_eval_every_episodes,
        "critic_eval_num_states": critic_eval_num_states,
        "hidden_layer_path": hidden_layer_path,
        "lr_recent_window": lr_recent_window,
        "lr_recent_saturate_at": lr_recent_saturate_at,
        "lr_recent_min_scale": lr_recent_min_scale,
        "lr_recent_power": lr_recent_power,
    }
    with open(os.path.join(run_dir, "run_config.json"), "w", encoding="utf-8") as fh:
        json.dump(run_config, fh, indent=2)

    env = gym.make("CartPole-v0")
    env.reset(seed=seed)

    # encoder shared by actor and critic
    encoder = InputLayer(rho=rho)
    n_inpt = encoder.n_inpt
    n_actions = 2

    # actor
    actor_net = build_actor_net(
        seed=seed,
        dt=dt,
        n_inpt=n_inpt,
        n_actions=n_actions,
        n_neurons_hid=n_neurons_hid,
        inhib_hid=inhib_hid,
        inhib_out=inhib_out,
        lr_hid=actor_lr_hid,
        lr_out=actor_lr_out,
        w_init_hid=actor_w_init_hid,
        w_init_out=actor_w_init_out,
        thresh_hid=actor_thresh_hid,
        thresh_out=actor_thresh_out,
        refrac_hid=actor_refrac_hid,
        refrac_out=actor_refrac_out,
        one_spike_hid=actor_one_spike_hid,
        one_spike_out=actor_one_spike_out,
        tc_trace_input=actor_tc_trace_input,
        tc_decay_hid=actor_tc_decay_hid,
        tc_decay_out=actor_tc_decay_out,
        tc_trace_hid=actor_tc_trace_hid,
        tc_trace_out=actor_tc_trace_out,
        tc_plus_out=actor_tc_plus_out,
        tc_minus_out=actor_tc_minus_out,
        tc_e_trace_out=actor_tc_e_trace_out,
        hidden_layer_path=hidden_layer_path,
    )

    actor_spikes = {}
    for layer in actor_net.layers:
        actor_spikes[layer] = Monitor(actor_net.layers[layer], state_vars=["s"], time=timesteps)
        actor_net.add_monitor(actor_spikes[layer], name=f"{layer}_spikes")

    # critic shares input encoding with actor
    state_to_shared_spikes = lambda s: build_grf_spike_train(
        encoder=encoder,
        state=s,
        sim_time=sim_time,
        dt=dt,
    )

    critic_net = SpikeCritic(
        n_inpt=n_inpt,
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
        wmax=critic_wmax,
        timesteps=timesteps,
        theta_plus=critic_theta_plus,
        tc_theta_decay=critic_tc_theta_decay,
    )

    # value_mean calibration
    if value_mean_mode == "bellman_centered":
        target_value = discounted_step_reward_fixed_point(
            step_reward=nonterminal_reward,
            gamma=gamma,
        )
        cal_states = sample_calibration_states(
            encoder=encoder,
            num_states=int(calibration_num_states),
            seed=seed + 7,
        )
        value_mean = calibrate_value_mean(
            critic_net=critic_net,
            state_to_spikes=state_to_shared_spikes,
            calibration_states=cal_states,
            sim_time=sim_time,
            value_std=value_std,
            gain=gain,
            target_value=target_value,
            reset_activity=critic_reset_activity,
        )
        print(f"[CALIB] Bellman-centered value_mean={value_mean:.4f} for target {target_value:.4f}")
    elif value_mean_mode == "manual":
        if value_mean is None:
            raise ValueError("value_mean must be provided when value_mean_mode='manual'")
        value_mean = float(value_mean)
    else:
        raise ValueError(f"Unknown value_mean_mode={value_mean_mode!r}")

    run_config["value_mean"] = value_mean
    with open(os.path.join(run_dir, "run_config.json"), "w", encoding="utf-8") as fh:
        json.dump(run_config, fh, indent=2)

    # validation set for critic diagnostics
    validation_states = None
    if critic_eval_num_states is not None and int(critic_eval_num_states) > 0:
        validation_states = sample_calibration_states(
            encoder=encoder,
            num_states=int(critic_eval_num_states),
            seed=seed + int(critic_eval_seed_offset),
        )
        pd.DataFrame(
            validation_states,
            columns=[f"s{i}" for i in range(validation_states.shape[1])],
        ).to_csv(os.path.join(run_dir, "critic_validation_states.csv"), index=False)

    critic_eval_records: List[Dict[str, object]] = []

    def record_critic_eval(tag: str, episode: int):
        if validation_states is None or len(validation_states) == 0:
            return
        metrics = evaluate_critic_validation_set(
            critic_net=critic_net,
            state_to_spikes=state_to_shared_spikes,
            validation_states=validation_states,
            sim_time=sim_time,
            value_mean=value_mean,
            value_std=value_std,
            gain=gain,
            reset_activity=critic_reset_activity,
        )
        metrics["episode"] = int(episode)
        metrics["tag"] = str(tag)
        critic_eval_records.append(metrics)
        pd.DataFrame(critic_eval_records).to_csv(
            os.path.join(run_dir, "critic_diagnostics.csv"), index=False,
        )
        print(
            f"[CRITIC-EVAL] ep={episode} | tag={tag} | "
            f"Vstd={metrics['value_std_eval']:.4f} | "
            f"Vrange={metrics['value_range_eval']:.4f} | "
            f"signed_rate_std={metrics['signed_rate_std_eval']:.5f} | "
            f"theta_max={metrics['theta_max_critic']:.4f}"
        )

    record_critic_eval(tag="init", episode=0)

    rewards_arr: List[float] = []
    ep_len_arr: List[int] = []
    terminations_arr: List[int] = []
    values_arr: List[float] = []
    td_errors_arr: List[float] = []
    spike_frac_vp_arr: List[float] = []
    active_vp_arr: List[float] = []

    best_window = 200
    best_mean_len = -np.inf  # for cartpole, larger ep_len is better
    best_ckpt_path: Optional[str] = None
    best_records: List[Dict[str, object]] = []
    best_log_path = os.path.join(run_dir, "best_checkpoints.csv")

    running_reward = 10.0

    for ep in tqdm(range(1, n_episodes + 1), desc="Train cartpole spike-critic"):
        episode_reset(actor_net)
        episode_reset(critic_net)

        # adaptive LR throttle based on recent completed episodes.
        # Same multiplier shared between actor and critic for the whole episode.
        lr_mult = recent_reward_lr_multiplier(
            rewards_arr,
            window=lr_recent_window,
            saturate_at=lr_recent_saturate_at,
            min_scale=lr_recent_min_scale,
            power=lr_recent_power,
        )
        critic_net.connections[("XF", "Vp")].nu = [0.0, critic_lr * lr_mult]

        state, _ = env.reset()
        ep_reward = 0.0
        terminated_flag = False
        ti = 0

        for ti in range(1, max_steps + 1):
            spk_X = build_grf_spike_train(encoder, state=state, sim_time=sim_time, dt=dt)
            actor_inpts_t = {"X": spk_X}

            # ---- actor evaluation pass on s_t ----
            actor_snap0 = store_state(actor_net)
            run_actor_pass(
                actor_net=actor_net,
                actor_inputs=actor_inpts_t,
                sim_time=sim_time,
                step=ti - 1,
                learning=False,
                reward=0.0,
                reset_activity=actor_reset_activity,
            )
            y_counts = get_spike_counts(actor_spikes, "Y")
            action_pred = int(torch.argmax(y_counts).item())
            action = e_greedy_action(action_pred, n_actions=n_actions, epsilon=epsilon)

            # ---- critic V(s_t) ----
            critic_snap0 = store_state(critic_net)
            out_s = run_critic_state(
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

            # ---- env step ----
            next_state, env_reward, terminated, truncated, _ = env.step(int(action))
            terminated_flag = bool(terminated_flag or terminated)

            if use_env_reward:
                reward_used = float(env_reward)
            else:
                reward_used = float(terminal_reward if terminated else nonterminal_reward)

            # ---- critic V(s_{t+1}) from same baseline ----
            restore_state(critic_net, critic_snap0)
            spk_X_next = build_grf_spike_train(encoder, state=next_state, sim_time=sim_time, dt=dt)
            out_ns = run_critic_state(
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

            # truncation is not an MDP terminal -> still bootstrap
            target = reward_used if terminated else (reward_used + gamma * V_ns)
            delta = float(target - V_s)

            # ---- actor learning pass on s_t ----
            restore_state(actor_net, actor_snap0)
            actor_reward = float(np.clip(delta / (value_std + 1e-8), -3.0, 3.0))
            scale = get_scale(running_reward, ep_reward, max_steps=max_steps)
            actor_net.connections[("H", "Y")].nu = [
                0.4 * scale * lr_mult * actor_lr_out,
                scale * lr_mult * actor_lr_out,
            ]
            run_actor_pass(
                actor_net=actor_net,
                actor_inputs=actor_inpts_t,
                sim_time=sim_time,
                step=ti - 1,
                learning=True,
                reward=actor_reward,
                reset_activity=actor_reset_activity,
            )

            # ---- critic online TD update on s_t ----
            restore_state(critic_net, critic_snap0)
            err_norm = float(np.clip(delta / (value_std + 1e-8), -3.0, 3.0))
            reward_dict = {critic_net.connections[("XF", "Vp")]: err_norm}
            _ = run_critic_state(
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

            ep_reward += reward_used
            values_arr.append(V_s)
            td_errors_arr.append(delta)
            spike_frac_vp_arr.append(out_s["spike_frac_vp"])
            active_vp_arr.append(out_s["active_neurons_vp"])

            state = next_state

            if terminated or truncated:
                break

        rewards_arr.append(ep_reward)
        ep_len_arr.append(ti)
        terminations_arr.append(int(terminated_flag))

        running_reward = 0.05 * ep_reward + 0.95 * running_reward
        if ep_reward >= 185:
            ckpt_path = os.path.join(run_dir, f"online_spike_critic_ep{ep}_prelim.pt")
            save_checkpoint(
                ckpt_path,
                actor_net,
                critic_net,
                extra={
                    "episode": ep,
                    "value_mean": value_mean,
                    "value_std": value_std,
                    "gain": gain,
                    "n_neurons_value": n_neurons_value,
                    "critic_lr": critic_lr,
                    "thresh_v": thresh_v,
                    "rewards_arr": list(rewards_arr),
                    **actor_time_constants,
                    **critic_time_constants,
                },
            )

        if ep % summary_every_episodes == 0:
            mean_len = summarize_last_window(ep_len_arr, best_window)
            theta_max_critic = theta_max_in_net(critic_net)
            print(
                f"EP={ep} | len={ti} | ep_reward={ep_reward:.2f} | "
                f"mean len({best_window})={mean_len:.2f} | "
                f"mean V({best_window})={summarize_last_window(values_arr, best_window):.4f} | "
                f"mean delta({best_window})={summarize_last_window(td_errors_arr, best_window):.4f} | "
                f"Vp sfrac={summarize_last_window(spike_frac_vp_arr, best_window):.4f} | "
                f"Vp active={summarize_last_window(active_vp_arr, best_window):.4f} | "
                f"theta_max_critic={theta_max_critic:.4f} | "
                f"lr_mult={lr_mult:.4f}"
            )

            if np.isfinite(mean_len) and mean_len > best_mean_len and len(ep_len_arr) >= best_window:
                best_mean_len = mean_len
                best_ckpt_path = os.path.join(run_dir, f"online_spike_critic_best_ep{ep}_lr_1e_4.pt")
                save_checkpoint(
                    best_ckpt_path,
                    actor_net,
                    critic_net,
                    extra={
                        "episode": ep,
                        "best_mean_len": best_mean_len,
                        "value_mean": value_mean,
                        "value_std": value_std,
                        "gain": gain,
                        "n_neurons_value": n_neurons_value,
                        "critic_lr": critic_lr,
                        "thresh_v": thresh_v,
                        "critic_w_init": critic_w_init,
                        "rewards_arr": list(rewards_arr),
                        **actor_time_constants,
                        **critic_time_constants,
                    },
                )
                best_records.append({
                    "episode": int(ep),
                    "best_mean_len": float(best_mean_len),
                    "window_size": int(best_window),
                    "checkpoint_path": str(best_ckpt_path),
                })
                write_checkpoint_log(best_log_path, best_records)

        if save_every_episodes is not None and save_every_episodes > 0 and ep % int(save_every_episodes) == 0:
            ckpt_path = os.path.join(run_dir, f"online_spike_critic_ep{ep}_lr_1e-4.pt")
            save_checkpoint(
                ckpt_path,
                actor_net,
                critic_net,
                extra={
                    "episode": ep,
                    "value_mean": value_mean,
                    "value_std": value_std,
                    "gain": gain,
                    "n_neurons_value": n_neurons_value,
                    "critic_lr": critic_lr,
                    "thresh_v": thresh_v,
                    "rewards_arr": list(rewards_arr),
                    **actor_time_constants,
                    **critic_time_constants,
                },
            )

        if critic_eval_every_episodes is not None and int(critic_eval_every_episodes) > 0:
            if ep % int(critic_eval_every_episodes) == 0:
                record_critic_eval(tag="periodic", episode=ep)

    if validation_states is not None and len(validation_states) > 0:
        if len(critic_eval_records) == 0 or critic_eval_records[-1]["episode"] != int(n_episodes):
            record_critic_eval(tag="final", episode=n_episodes)

    env.close()

    pd.DataFrame({
        "episode": np.arange(1, len(ep_len_arr) + 1),
        "ep_reward": rewards_arr,
        "ep_len": ep_len_arr,
        "terminated": terminations_arr,
    }).to_csv(os.path.join(run_dir, "episode_history.csv"), index=False)

    summary = {
        "run_name": run_name if run_name is not None else Path(run_dir).name,
        "run_dir": run_dir,
        "episodes_completed": int(len(ep_len_arr)),
        "best_mean_len": float(best_mean_len) if np.isfinite(best_mean_len) else np.nan,
        "mean_len_last_200": summarize_last_window(ep_len_arr, 200),
        "mean_reward_last_200": summarize_last_window(rewards_arr, 200),
        "value_last_200": summarize_last_window(values_arr, 200),
        "delta_last_200": summarize_last_window(td_errors_arr, 200),
        "critic_spike_frac_last_200": summarize_last_window(spike_frac_vp_arr, 200),
        "critic_active_last_200": summarize_last_window(active_vp_arr, 200),
        "theta_max_critic_final": theta_max_in_net(critic_net),
        "theta_max_actor_final": theta_max_in_net(actor_net),
        "best_ckpt_path": best_ckpt_path,
        "value_mean": value_mean,
        "value_std": value_std,
        "gain": gain,
        "n_neurons_value": n_neurons_value,
        "critic_lr": critic_lr,
        "actor_lr_out": actor_lr_out,
        **actor_time_constants,
        **critic_time_constants,
    }
    with open(os.path.join(run_dir, "run_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    if plot_results:
        plt.figure()
        plt.plot(ep_len_arr)
        plt.title("Episode length")
        plt.grid(True)
        plt.show()

        plt.figure()
        plt.plot(rewards_arr)
        plt.title("Episode reward")
        plt.grid(True)
        plt.show()

        if len(values_arr) > 0:
            plt.figure()
            plt.plot(values_arr[-2000:])
            plt.title("V(s) last 2000 steps")
            plt.grid(True)
            plt.show()

        if len(td_errors_arr) > 0:
            plt.figure()
            plt.plot(td_errors_arr[-2000:])
            plt.title("TD error last 2000 steps")
            plt.grid(True)
            plt.show()

        plt.figure()
        plt.plot(pd.Series(spike_frac_vp_arr).rolling(100, min_periods=1).mean(), label="Vp spike frac")
        plt.title("Critic spike fraction rolling-100")
        plt.legend()
        plt.grid(True)
        plt.show()

    return summary


if __name__ == "__main__":
    main(
        seed=0,
        n_episodes=100_000,
        max_steps=200,
        time_sec=20.0,
        dt=0.5,
        gamma=0.99,
        epsilon=0.0,

        # actor (500-neuron pretrained hidden layer)
        rho=5000.0,
        n_neurons_hid=500,
        inhib_hid=0.1,
        inhib_out=0.1,
        actor_lr_hid=1e-5,
        actor_lr_out=1e-2,
        actor_w_init_hid=0.5,
        actor_w_init_out=0.5,
        actor_thresh_hid=-58.0,
        actor_thresh_out=-63.0,
        actor_refrac_hid=5,
        actor_refrac_out=0,
        actor_one_spike_hid=False,
        actor_one_spike_out=True,
        actor_reset_activity=True,
        hidden_layer_path=HIDDEN_LAYER_PATH,

        # critic
        n_neurons_value=10,
        critic_lr=2e-4,
        thresh_v=-63.0,
        critic_w_init=0.2,
        gain=4.0,
        critic_tc_trace_v=30.0,
        critic_tc_decay_v=80.0,
        critic_tc_plus=30.0,
        critic_tc_minus=30.0,
        critic_tc_e_trace=30.0,
        critic_theta_plus=0.0,
        critic_tc_theta_decay=10.0,
        critic_reset_activity=False,

        # value readout: V = value_mean + value_std * gain * rate_p
        value_mean=None,
        value_std=50.0,
        value_mean_mode="bellman_centered",
        calibration_num_states=128,
        nonterminal_reward=1.0,
        terminal_reward=0.0,
        use_env_reward=True,

        # adaptive LR throttle on both actor & critic, keyed on recent rewards.
        # With saturate_at=180 and power=2: mean_recent=90 -> 0.25,
        # mean_recent=150 -> 0.025, mean_recent>=180 -> floor 0.01.
        lr_recent_window=10,
        lr_recent_saturate_at=180.0,
        lr_recent_min_scale=0.01,
        lr_recent_power=2.0,

        summary_every_episodes=25,
        save_every_episodes=500,
        critic_eval_every_episodes=100,
        critic_eval_num_states=128,
        plot_results=True,
    )

    print("ok")
