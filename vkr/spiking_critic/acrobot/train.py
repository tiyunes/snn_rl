"""
2-layer spiking actor + spiking GRF critic for Acrobot-v1, with a 100-neuron H layer.

Hidden layer checkpoint: hidden_layers/acrobot_100n/hidden_layer.pt

Per-step 4-pass structure:
  Pass 1: actor(s_t, no-learn)  -> get action
  Pass 2: critic(s_t, no-learn) -> V_s
  Pass 3: critic(s_{t+1}, no-learn) from same critic snapshot -> V_ns
  Pass 4a: restore actor + actor(s_t, learn) with reward = clip(delta/value_std, -3, 3)
  Pass 4b: restore critic + critic(s_t, learn) with reward_dict = {conn: err_norm}
"""

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, REPO_ROOT)  # local modified bindsnet

import ast
import json
import random

import numpy as np
import pandas as pd
import torch
import gymnasium as gym

from time import time as now
from typing import Optional, List

from bindsnet.encoding import poisson
from bindsnet.learning import MSTDPET
from bindsnet.network import Network
from bindsnet.network.nodes import Input, DiehlAndCookNodesModified
from bindsnet.network.topology import Connection
from bindsnet.network.monitors import Monitor
from bindsnet.models import Diehl2LayersWTAOutput


# =========================================================
# PATHS
# =========================================================
DATA_PATH = os.path.join(REPO_ROOT, "data", "acrobot_state_action.csv")
HIDDEN_LAYER_PATH = os.path.join(REPO_ROOT, "hidden_layers", "acrobot_100n", "hidden_layer.pt")
ACTOR_SAVE_DIR = os.path.join(SCRIPT_DIR, "runs")


def find_best_hidden_checkpoint(run_dir: str) -> str:
    """Pick the epoch with the minimum env mean_length from metrics.json.

    Falls back to final.pt if metrics.json is missing.
    """
    metrics_path = os.path.join(run_dir, "metrics.json")
    if not os.path.exists(metrics_path):
        fallback = os.path.join(run_dir, "final.pt")
        print(f"[H-LAYER] metrics.json not found — using {fallback}")
        return fallback

    with open(metrics_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    epochs = data.get("epochs", [])
    if not epochs:
        fallback = os.path.join(run_dir, "final.pt")
        print(f"[H-LAYER] metrics.json has no epochs — using {fallback}")
        return fallback

    best = min(epochs, key=lambda e: e["mean_length"])
    ckpt = os.path.join(run_dir, f"epoch_{best['epoch']}.pt")
    print(
        f"[H-LAYER] best epoch={best['epoch']}  mean_length={best['mean_length']:.1f}  "
        f"success={best['success_rate']:.2f}  ret={best['mean_return']:.1f}"
    )
    print(f"[H-LAYER] using {ckpt}")
    return ckpt


# =========================================================
# GRF ENCODER
# =========================================================
class GRFInputLayer:
    def __init__(
        self,
        centers_per_feature=(12, 12, 12, 12, 10, 10),
        feature_ranges=None,
        sigma_scale: float = 0.6,
        rho: float = 500.0,
        normalize_per_feature: bool = True,
        clip_rates=None,
        device: str = "cpu",
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
            self.centers.append(c)
            self.sigmas.append(np.full_like(c, max(self.sigma_scale * spacing, 1e-6)))
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


def build_grf_spike_train(enc: GRFInputLayer, state, sim_time: float, dt: float) -> torch.Tensor:
    """Returns [timesteps, 1, n_inpt] Poisson spike train for the given state."""
    rates = torch.from_numpy(enc.encode_numpy(np.asarray(state, dtype=np.float32)))
    spk = poisson(datum=rates, time=sim_time, dt=dt)
    if spk.ndim == 2:
        spk = spk.unsqueeze(1)
    return spk.float()


# =========================================================
# RESET HELPERS
# =========================================================
def reset_activity_only(net):
    for layer in net.layers.values():
        for attr in ("v", "s", "x", "refrac_count"):
            t = getattr(layer, attr, None)
            if torch.is_tensor(t):
                if attr == "v":
                    t.fill_(float(getattr(layer, "rest", 0.0)))
                else:
                    t.zero_()
    for mon in net.monitors.values():
        mon.reset_state_variables()


def episode_reset(net):
    """Episode-boundary reset: preserve theta, zero activity + MSTDPET short-term traces."""
    for layer in net.layers.values():
        for attr in ("v", "s", "x", "refrac_count"):
            t = getattr(layer, attr, None)
            if torch.is_tensor(t):
                if attr == "v":
                    t.fill_(float(getattr(layer, "rest", 0.0)))
                else:
                    t.zero_()
    for conn in net.connections.values():
        rule = conn.update_rule
        for attr in ("eligibility_trace", "p_plus", "p_minus", "eligibility"):
            t = getattr(rule, attr, None)
            if torch.is_tensor(t):
                try:
                    t.zero_()
                except Exception:
                    pass
    for mon in net.monitors.values():
        mon.reset_state_variables()


def _clone_if_tensor(obj, attr):
    t = getattr(obj, attr, None)
    return t.clone().detach() if torch.is_tensor(t) else None


def store_state(net) -> dict:
    state = {"layers": {}, "rules": {}}
    for lname, layer in net.layers.items():
        d = {}
        for attr in ("v", "s", "x", "refrac_count", "theta", "decay"):
            t = _clone_if_tensor(layer, attr)
            if t is not None:
                d[attr] = t
        state["layers"][lname] = d
    for ck, conn in net.connections.items():
        rule = conn.update_rule
        d = {}
        for attr in ("p_plus", "p_minus", "eligibility", "eligibility_trace"):
            t = _clone_if_tensor(rule, attr)
            if t is not None:
                d[attr] = t
        if d:
            state["rules"][ck] = d
    return state


def restore_state(net, state: dict):
    for lname, d in state["layers"].items():
        if lname not in net.layers:
            continue
        layer = net.layers[lname]
        for attr, t in d.items():
            cur = getattr(layer, attr, None)
            if torch.is_tensor(cur):
                cur.copy_(t)
    for ck, d in state["rules"].items():
        if ck not in net.connections:
            continue
        rule = net.connections[ck].update_rule
        for attr, t in d.items():
            cur = getattr(rule, attr, None)
            if torch.is_tensor(cur):
                cur.copy_(t)


# =========================================================
# SPIKING CRITIC
# =========================================================
class SpikeCritic(Network):
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
        timesteps: int = 40,
        theta_plus: float = 0.0,
        tc_theta_decay: float = 1e4,
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
def read_value_components(critic: SpikeCritic):
    Vp = critic.layers["Vp"]
    tc_p = float(Vp.tc_trace.item() if torch.is_tensor(Vp.tc_trace) else Vp.tc_trace)
    rate_p = float((Vp.x.mean(dim=1) / tc_p)[0].item())
    return rate_p, 0.0, rate_p


def signed_rate_to_value(signed_rate: float, value_mean: float, value_std: float, gain: float) -> float:
    return float(value_mean + value_std * gain * signed_rate)


# =========================================================
# HELPERS
# =========================================================
def e_greedy_action(best_action: int, epsilon: float, n_actions: int = 3) -> int:
    prob = np.full(n_actions, epsilon / n_actions)
    prob[best_action] += 1.0 - epsilon
    return int(np.random.choice(n_actions, p=prob))


def lr_scale(running_reward: float, lr_out: float) -> float:
    t = max(0.0, min(1.0, (running_reward - (-500.0)) / ((-100.0) - (-500.0))))
    return lr_out * (1.0 - 0.9 * t)


def calibrate_value_mean(
    critic: SpikeCritic,
    enc: GRFInputLayer,
    states_np: np.ndarray,
    sim_time: float,
    dt: float,
    value_std: float,
    gain: float,
    target_value: float,
    n_calib: int = 128,
    seed: int = 0,
) -> float:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(states_np), size=min(n_calib, len(states_np)), replace=False)
    signed_rates = []
    for state in states_np[idx]:
        reset_activity_only(critic)
        spk = build_grf_spike_train(enc, state, sim_time, dt)
        with torch.no_grad():
            critic.run(inputs={"XF": spk}, time=sim_time, reward=0.0, step=0, one_step=False, learning=False)
        _, _, sr = read_value_components(critic)
        signed_rates.append(sr)
    mean_sr = float(np.mean(signed_rates))
    value_mean = float(target_value - value_std * gain * mean_sr)
    print(
        f"[CALIB] mean_signed_rate={mean_sr:.5f}  target_value={target_value:.2f}  "
        f"value_mean={value_mean:.4f}"
    )
    return value_mean


# =========================================================
# MAIN
# =========================================================
def main(
    seed: int = 0,
    n_iter: int = 7000,
    # actor architecture
    n_neurons_hid: int = 100,
    n_neurons_out: int = 3,
    inhib_hid: float = 0.1,
    inhib_out: float = 0.1,
    lr_hid: float = 1e-5,
    lr_out: float = 1e-4,
    time: int = 20,
    dt: float = 0.5,
    theta_plus: float = 0.0,
    tc_theta_decay: float = 1e7,
    w_init_hid: float = 0.7,
    w_init_out: float = 0.5,
    thresh_hid: float = -53.0,
    thresh_out: float = -64.0,
    refrac_hid: int = 0,
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
    # GRF encoder
    centers_per_feature: tuple = (12, 12, 12, 12, 10, 10),
    sigma_scale: float = 0.6,
    rho: float = 500.0,
    normalize_per_feature: bool = True,
    q_low: float = 0.01,
    q_high: float = 0.99,
    # critic architecture
    n_neurons_value: int = 30,
    critic_lr: float = 2e-6,
    thresh_v: float = -59.0,
    tc_trace_v: float = 30.0,
    tc_decay_v: float = 80.0,
    tc_trace_input_critic: float = 20.0,
    tc_plus_critic: float = 30.0,
    tc_minus_critic: float = 30.0,
    tc_e_trace_critic: float = 30.0,
    critic_w_init: float = 0.2,
    critic_wmax: float = 1.0,
    critic_theta_plus: float = 0.0,
    critic_tc_theta_decay: float = 1e4,
    # value readout and calibration
    gain: float = 4.0,
    value_std: float = 50.0,
    nonterminal_reward: float = -1.0,
    n_calib: int = 128,
    # RL
    gamma: float = 0.99,
    epsilon: float = 0.1,
    save_every: int = 50,
    resume_tag: Optional[str] = None,
    hidden_layer_path: Optional[str] = None,
):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    os.makedirs(ACTOR_SAVE_DIR, exist_ok=True)
    timesteps = int(round(time / dt))

    if hidden_layer_path is None:
        hidden_layer_path = HIDDEN_LAYER_PATH
    if not os.path.exists(hidden_layer_path):
        raise FileNotFoundError(f"H-layer checkpoint not found: {hidden_layer_path}")

    df = pd.read_csv(DATA_PATH)
    state_arr = np.array(df["state"].apply(parse_state_column).tolist(), dtype=np.float32)
    feature_ranges = build_feature_ranges_from_data(state_arr, q_low=q_low, q_high=q_high)

    enc = GRFInputLayer(
        centers_per_feature=centers_per_feature,
        feature_ranges=feature_ranges,
        sigma_scale=sigma_scale,
        rho=rho,
        normalize_per_feature=normalize_per_feature,
    )
    print(f"Encoder n_inpt = {enc.n_inpt}")

    if resume_tag is not None:
        _actor_path = os.path.join(ACTOR_SAVE_DIR, f"actor_{resume_tag}.pt")
        if not os.path.exists(_actor_path):
            print(f"[WARN] Checkpoint '{resume_tag}' not found ({_actor_path}), starting fresh.")
            resume_tag = None

    if resume_tag is not None:
        actor  = torch.load(os.path.join(ACTOR_SAVE_DIR, f"actor_{resume_tag}.pt"))
        critic = torch.load(os.path.join(ACTOR_SAVE_DIR, f"critic_{resume_tag}.pt"))
        ckpt   = torch.load(os.path.join(ACTOR_SAVE_DIR, f"state_{resume_tag}.pt"))

        rewards_arr      = list(ckpt.get("rewards_arr", []))
        ep_lengths_arr   = list(ckpt.get("ep_lengths_arr", []))
        start_ep         = int(ckpt.get("ep", len(rewards_arr)))
        value_mean       = float(ckpt["value_mean"])
        value_std        = float(ckpt.get("value_std", value_std))
        gain             = float(ckpt.get("gain", gain))
        running_reward   = float(ckpt.get("running_reward",
                                          rewards_arr[-1] if rewards_arr else -500.0))
        best_ep_length   = float(min(ep_lengths_arr)) if ep_lengths_arr else float("inf")
        values_arr       = []
        td_errors_arr    = []
        step_rewards_arr = []

        print(f"[RESUME] Loaded '{resume_tag}': start_ep={start_ep}, "
              f"best_ep_length={best_ep_length:.0f}, running_reward={running_reward:.1f}")
    else:
        actor = Diehl2LayersWTAOutput(
            n_inpt=enc.n_inpt,
            n_neurons_hid=n_neurons_hid,
            n_neurons_out=n_neurons_out,
            inh_hid=inhib_hid,
            inh_out=inhib_out,
            dt=dt,
            nu_hid=[0.25 * lr_hid, lr_hid],
            nu_out=[0.25 * lr_out, lr_out],
            wmin=0.0,
            wmax=1.0,
            norm=None,
            theta_plus=theta_plus,
            tc_theta_decay=tc_theta_decay,
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

        critic = SpikeCritic(
            n_inpt=enc.n_inpt,
            n_neurons_value=n_neurons_value,
            dt=dt,
            critic_lr=critic_lr,
            thresh_v=thresh_v,
            tc_trace_v=tc_trace_v,
            tc_decay_v=tc_decay_v,
            tc_trace_input=tc_trace_input_critic,
            tc_plus=tc_plus_critic,
            tc_minus=tc_minus_critic,
            tc_e_trace=tc_e_trace_critic,
            w_init=critic_w_init,
            wmax=critic_wmax,
            timesteps=timesteps,
            theta_plus=critic_theta_plus,
            tc_theta_decay=critic_tc_theta_decay,
        )

        target_value = nonterminal_reward / (1.0 - gamma)
        value_mean = calibrate_value_mean(
            critic=critic,
            enc=enc,
            states_np=state_arr,
            sim_time=time,
            dt=dt,
            value_std=value_std,
            gain=gain,
            target_value=target_value,
            n_calib=n_calib,
            seed=seed,
        )

        start_ep         = 0
        running_reward   = -500.0
        rewards_arr      = []
        ep_lengths_arr   = []
        values_arr       = []
        td_errors_arr    = []
        step_rewards_arr = []
        best_ep_length   = float("inf")

    spk_mon = {}
    for lname in actor.layers:
        spk_mon[lname] = Monitor(actor.layers[lname], state_vars=["s"], time=timesteps)
        actor.add_monitor(spk_mon[lname], name=f"{lname}_spikes")

    env = gym.make("Acrobot-v1", render_mode="human")

    def _save_checkpoint(tag: str):
        actor.save(os.path.join(ACTOR_SAVE_DIR, f"actor_{tag}.pt"))
        critic.save(os.path.join(ACTOR_SAVE_DIR, f"critic_{tag}.pt"))
        torch.save(
            {
                "ep_lengths_arr": ep_lengths_arr,
                "rewards_arr": rewards_arr,
                "ep": len(rewards_arr),
                "value_mean": value_mean,
                "value_std": value_std,
                "gain": gain,
                "running_reward": running_reward,
                "hidden_layer_path": hidden_layer_path,
                "n_neurons_hid": n_neurons_hid,
            },
            os.path.join(ACTOR_SAVE_DIR, f"state_{tag}.pt"),
        )

    end_ep = start_ep + n_iter
    print(f"Starting RL training: ep {start_ep} -> {end_ep}, "
          f"n_neurons_hid={n_neurons_hid}, n_neurons_value={n_neurons_value}")
    t0 = now()

    for ep in range(start_ep, end_ep):
        obs, _ = env.reset(seed=seed + ep)
        state = np.array(obs, dtype=np.float32)
        ep_reward = 0.0
        episode_reset(actor)
        episode_reset(critic)

        for ti in range(1, 1001):
            spk_X = build_grf_spike_train(enc, state, time, dt)

            actor_snap = store_state(actor)
            actor.run(inputs={"X": spk_X}, time=time, reward=0.0,
                      step=ti - 1, one_step=False, learning=False)
            y_counts = spk_mon["Y"].get("s").sum(dim=0)[0]
            action = e_greedy_action(int(y_counts.argmax()), epsilon, n_neurons_out)

            critic_snap = store_state(critic)
            critic.run(inputs={"XF": spk_X}, time=time, reward=0.0,
                       step=0, one_step=False, learning=False)
            _, _, sr_s = read_value_components(critic)
            V_s = signed_rate_to_value(sr_s, value_mean, value_std, gain)

            next_obs, env_reward, terminated, truncated, _ = env.step(action)
            next_state = np.array(next_obs, dtype=np.float32)
            shaped_reward = 10.0 if terminated else nonterminal_reward
            ep_reward += env_reward

            restore_state(critic, critic_snap)
            spk_X_next = build_grf_spike_train(enc, next_state, time, dt)
            critic.run(inputs={"XF": spk_X_next}, time=time, reward=0.0,
                       step=0, one_step=False, learning=False)
            _, _, sr_ns = read_value_components(critic)
            V_ns = signed_rate_to_value(sr_ns, value_mean, value_std, gain)

            target = shaped_reward if terminated else shaped_reward + gamma * V_ns
            delta = target - V_s

            values_arr.append(V_s)
            td_errors_arr.append(delta)
            step_rewards_arr.append(shaped_reward)

            actor.connections[("H", "Y")].nu = lr_scale(running_reward, lr_out)
            restore_state(actor, actor_snap)
            actor_reward = float(np.clip(delta / (value_std + 1e-8), -3.0, 3.0))
            actor.run(inputs={"X": spk_X}, time=time, reward=actor_reward,
                      step=ti - 1, one_step=False, learning=True)

            restore_state(critic, critic_snap)
            err_norm = float(np.clip(delta / (value_std + 1e-8), -3.0, 3.0))
            reward_dict = {critic.connections[("XF", "Vp")]: err_norm}
            critic.run(inputs={"XF": spk_X}, time=time, reward=reward_dict,
                       step=0, one_step=False, learning=True)

            state = next_state
            if terminated:
                break

        ep_length = ti
        running_reward = 0.05 * ep_reward + 0.95 * running_reward
        rewards_arr.append(ep_reward)
        ep_lengths_arr.append(ep_length)

        if ep_length < best_ep_length:
            best_ep_length = ep_length
            _save_checkpoint("best")
            print(f"  -> new best ep_length={ep_length} at ep {ep + 1}, saved")

        if (ep + 1) % 50 == 0:
            elapsed = now() - t0
            print(
                f"[ep {ep + 1:5d}/{end_ep}]  ep_reward={ep_reward:7.1f}  "
                f"ep_length={ep_length:5d}  running={running_reward:7.1f}  "
                f"elapsed={elapsed / 60:.1f} min"
            )

        if (ep + 1) % save_every == 0:
            _save_checkpoint(f"ep{ep + 1}")
            print(f"  -> saved checkpoint ep{ep + 1}")

    env.close()
    _save_checkpoint("final")
    print(f"\nTraining done. Saved to {ACTOR_SAVE_DIR}")
    print(f"Total time: {(now() - t0) / 60:.1f} min")
    print(f"Best episode length: {best_ep_length}")

    return rewards_arr, ep_lengths_arr


if __name__ == "__main__":
    seed = 0
    n_iter = 15_000

    n_neurons_hid = 100  # must match the pretrained H-layer checkpoint
    n_neurons_out = 3

    inhib_hid = 0.1
    inhib_out = 0.1

    lr_hid = 1e-3   # H is frozen (NoOp), unused
    lr_out = 1e-3

    time_ = 20
    dt = 0.5

    theta_plus = 0.0
    tc_theta_decay = 1e7

    w_init_hid = 0.7  # not used (H weights loaded from checkpoint)
    w_init_out = 0.8

    thresh_hid = -53.0
    thresh_out = -64.0

    refrac_hid = 0
    refrac_out = 0

    one_spike_hid = False
    one_spike_out = True

    tc_trace_input = 20.0
    tc_decay_hid = 15.0
    tc_decay_out = 15.0
    tc_trace_hid = 10.0
    tc_trace_out = 10.0
    tc_plus_out = 10.0
    tc_minus_out = 10.0
    tc_e_trace_out = 15.0

    centers_per_feature = (12, 12, 12, 12, 10, 10)
    sigma_scale = 0.6
    rho = 500.0
    normalize_per_feature = True

    n_neurons_value = 30
    critic_lr = 1e-3
    thresh_v = -63.0
    tc_trace_v = 30.0
    tc_decay_v = 80.0
    tc_trace_input_critic = 20.0
    tc_plus_critic = 30.0
    tc_minus_critic = 30.0
    tc_e_trace_critic = 30.0
    critic_w_init = 0.2
    critic_wmax = 1.0
    critic_theta_plus = 0.0
    critic_tc_theta_decay = 1e4

    gain = 4.0
    value_std = 50.0
    nonterminal_reward = -1.0
    n_calib = 128

    gamma = 0.99
    epsilon = 0.0

    resume_tag = None  # e.g. "ep1600" to continue from runs/actor_ep1600.pt

    hidden_layer_path = HIDDEN_LAYER_PATH

    rewards, ep_lengths = main(
        seed=seed,
        n_iter=n_iter,
        n_neurons_hid=n_neurons_hid,
        n_neurons_out=n_neurons_out,
        inhib_hid=inhib_hid,
        inhib_out=inhib_out,
        lr_hid=lr_hid,
        lr_out=lr_out,
        time=time_,
        dt=dt,
        theta_plus=theta_plus,
        tc_theta_decay=tc_theta_decay,
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
        centers_per_feature=centers_per_feature,
        sigma_scale=sigma_scale,
        rho=rho,
        normalize_per_feature=normalize_per_feature,
        n_neurons_value=n_neurons_value,
        critic_lr=critic_lr,
        thresh_v=thresh_v,
        tc_trace_v=tc_trace_v,
        tc_decay_v=tc_decay_v,
        tc_trace_input_critic=tc_trace_input_critic,
        tc_plus_critic=tc_plus_critic,
        tc_minus_critic=tc_minus_critic,
        tc_e_trace_critic=tc_e_trace_critic,
        critic_w_init=critic_w_init,
        critic_wmax=critic_wmax,
        critic_theta_plus=critic_theta_plus,
        critic_tc_theta_decay=critic_tc_theta_decay,
        gain=gain,
        value_std=value_std,
        nonterminal_reward=nonterminal_reward,
        n_calib=n_calib,
        gamma=gamma,
        epsilon=epsilon,
        save_every=50,
        resume_tag=resume_tag,
        hidden_layer_path=hidden_layer_path,
    )

    print("ok")
