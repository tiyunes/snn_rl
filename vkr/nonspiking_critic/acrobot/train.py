"""
2-layer spiking actor + non-spiking Fourier-feature critic for Acrobot-v1.

Architecture:
  X (68 GRF inputs) -> H (20 neurons, frozen pretrained) -> Y (3 output neurons, MSTDPET)
  Critic: linear value function over order-2 Fourier features (non-spiking)

Hidden layer checkpoint: hidden_layers/acrobot_20n/hidden_layer.pt

  - GRFInputLayer (compact, 68 neurons)
  - Feature ranges inferred from dataset (same as during hidden-layer pretraining)
  - Monitor buffer = timesteps (40) to capture all spikes per step
"""

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, REPO_ROOT)  # local modified bindsnet

import ast
import math
import random
import itertools

import numpy as np
import pandas as pd
import torch
import gymnasium as gym

from time import time as now
from typing import Optional, List, Tuple

from bindsnet.encoding import poisson
from bindsnet.network.monitors import Monitor
from bindsnet.models import Diehl2LayersWTAOutput


# =========================================================
# PATHS
# =========================================================
DATA_PATH = os.path.join(REPO_ROOT, "data", "acrobot_state_action.csv")
HIDDEN_LAYER_PATH = os.path.join(REPO_ROOT, "hidden_layers", "acrobot_20n", "hidden_layer.pt")
ACTOR_SAVE_DIR = os.path.join(SCRIPT_DIR, "runs")


# =========================================================
# GRF ENCODER  (matches hidden_layers/acrobot_20n/train_hidden.py)
# =========================================================
class GRFInputLayer:
    def __init__(
        self,
        centers_per_feature=(12, 12, 12, 12, 10, 10),
        feature_ranges=None,
        sigma_scale: float = 0.6,
        rho: float = 500.0,
        normalize_per_feature: bool = True,
        clip_rates: Optional[Tuple[float, float]] = None,
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
        return torch.tensor(self.encode_numpy(np.asarray(state, dtype=np.float32)),
                            dtype=torch.float32, device=self.device)


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


# =========================================================
# ACTOR STATE SAVE / RESTORE
# =========================================================
def store_actor_state(net):
    state = {
        'X': {'x': net.layers['X'].x.clone().detach()},
        'H': {
            'v': net.layers['H'].v.clone().detach(),
            'x': net.layers['H'].x.clone().detach(),
            'refrac_count': net.layers['H'].refrac_count.clone().detach(),
            'decay': net.layers['H'].decay.clone().detach(),
        },
        'Y': {
            'v': net.layers['Y'].v.clone().detach(),
            'x': net.layers['Y'].x.clone().detach(),
            'refrac_count': net.layers['Y'].refrac_count.clone().detach(),
            'decay': net.layers['Y'].decay.clone().detach(),
        },
        'HY': {'p_plus': None, 'p_minus': None, 'eligibility': None, 'eligibility_trace': None},
    }
    rule = net.connections[('H', 'Y')].update_rule
    for k in ('p_plus', 'p_minus', 'eligibility', 'eligibility_trace'):
        v = getattr(rule, k, None)
        if v is not None:
            state['HY'][k] = v.clone().detach()
    return state


def restore_actor_state(net, state):
    net.layers['X'].x.copy_(state['X']['x'])
    for layer_key in ('H', 'Y'):
        for attr in ('v', 'x', 'refrac_count', 'decay'):
            getattr(net.layers[layer_key], attr).copy_(state[layer_key][attr])
    rule = net.connections[('H', 'Y')].update_rule
    for k in ('p_plus', 'p_minus', 'eligibility', 'eligibility_trace'):
        if state['HY'][k] is not None:
            getattr(rule, k).copy_(state['HY'][k])


# =========================================================
# HELPERS
# =========================================================
def e_greedy_action(best_action: int, epsilon: float, n_actions: int = 3) -> int:
    prob = np.full(n_actions, epsilon / n_actions)
    prob[best_action] += 1.0 - epsilon
    return int(np.random.choice(n_actions, p=prob))


def fourier_features(state: np.ndarray, combns: np.ndarray) -> np.ndarray:
    lims = [(-1., 1.), (-1., 1.), (-1., 1.), (-1., 1.), (-12.567, 12.567), (-28.274, 28.274)]
    norm = np.array([(state[j] - lo) / (hi - lo) for j, (lo, hi) in enumerate(lims)], dtype=np.float64)
    return np.cos(np.pi * combns.dot(norm))


def lr_scale(running_reward: float, lr_out: float) -> float:
    """Adaptive scale: larger updates when performance is poor."""
    worst, best = -500.0, -100.0
    t = max(0.0, min(1.0, (running_reward - worst) / (best - worst)))
    return lr_out * (1.0 - 0.9 * t)   # lr_out .. 0.1*lr_out


# =========================================================
# MAIN
# =========================================================
def main(
    seed: int = 0,
    n_iter: int = 7000,
    n_neurons_hid: int = 20,
    n_neurons_out: int = 3,
    inhib_hid: float = 0.1,
    inhib_out: float = 0.1,
    lr_hid: float = 1e-5,
    lr_out: float = 2e-3,
    time: int = 20,
    dt: float = 0.5,
    theta_plus: float = 0.0,
    tc_theta_decay: float = 1e7,
    w_init_hid: float = 0.7,
    w_init_out: float = 0.5,
    thresh_hid: float = -53.0,
    thresh_out: float = -55.0,
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
    gamma: float = 0.99,
    lda: float = 0.5,
    alpha: float = 0.001,
    epsilon: float = 0.1,
    centers_per_feature: tuple = (12, 12, 12, 12, 10, 10),
    sigma_scale: float = 0.6,
    rho: float = 500.0,
    normalize_per_feature: bool = True,
    q_low: float = 0.01,
    q_high: float = 0.99,
    save_every: int = 50,
):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    os.makedirs(ACTOR_SAVE_DIR, exist_ok=True)

    timesteps = int(round(time / dt))  # 40 steps = 20 ms / 0.5 ms

    # --- Feature ranges from pretraining dataset ---
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

    # --- Actor network ---
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
    )

    # Monitors: buffer = timesteps to capture all spikes
    spk_mon = {}
    for layer in actor.layers:
        spk_mon[layer] = Monitor(actor.layers[layer], state_vars=['s'], time=timesteps)
        actor.add_monitor(spk_mon[layer], name=f'{layer}_spikes')

    # --- Non-spiking Fourier critic ---
    order = 2
    combns = np.array(list(itertools.product(range(order + 1), repeat=6)))
    w_v = np.zeros(len(combns), dtype=np.float64)

    env = gym.make('Acrobot-v1', render_mode="human")
    running_reward = -500.0

    # --- Per-episode tracking ---
    rewards_arr: list = []
    ep_lengths_arr: list = []

    # --- Per-step tracking ---
    values_arr: list = []       # v_s at each step
    td_errors_arr: list = []    # delta_t at each step
    step_rewards_arr: list = [] # shaped_reward at each step

    best_ep_length = float('inf')

    def _save_checkpoint(tag: str):
        actor.save(os.path.join(ACTOR_SAVE_DIR, f"actor_{tag}.pt"))
        torch.save(
            {
                "w_v": w_v,
                "ep_lengths_arr": ep_lengths_arr,
                "rewards_arr": rewards_arr,
                "values_arr": values_arr,
                "td_errors_arr": td_errors_arr,
                "step_rewards_arr": step_rewards_arr,
                "ep": len(rewards_arr),
            },
            os.path.join(ACTOR_SAVE_DIR, f"state_{tag}.pt"),
        )

    print(f"Starting RL training: {n_iter} episodes")
    t0 = now()

    for ep in range(n_iter):
        obs, _ = env.reset(seed=seed + ep)
        state = np.array(obs, dtype=np.float32)
        ep_reward = 0.0
        saved_actor_state = None
        e_v = np.zeros(len(combns), dtype=np.float64)

        for ti in range(1, 1001):
            rates = enc.step(state)
            spk_X = poisson(datum=rates, time=time, dt=dt)
            phi_s = fourier_features(state, combns)

            # === First pass: learning=False, get action ===
            if ti > 1:
                saved_actor_state = store_actor_state(actor)

            actor.run(inputs={'X': spk_X}, time=time, reward=0.0,
                      one_step=False, step=ti - 1, learning=False)

            y_counts = spk_mon['Y'].get('s').sum(dim=0)[0]  # [n_neurons_out]
            best_action = int(y_counts.argmax().item())
            action = e_greedy_action(best_action, epsilon, n_neurons_out)

            # === Step environment ===
            next_obs, env_reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = np.array(next_obs, dtype=np.float32)

            shaped_reward = 10.0 if terminated else -1.0

            # === Critic TD update ===
            phi_ns = fourier_features(next_state, combns)
            e_v = gamma * lda * e_v + phi_s
            v_s = float(np.dot(w_v, phi_s))
            v_ns = 0.0 if done else float(np.dot(w_v, phi_ns))
            delta_t = shaped_reward + gamma * v_ns - v_s
            w_v += alpha * delta_t * e_v

            # Record per-step diagnostics
            values_arr.append(v_s)
            td_errors_arr.append(delta_t)
            step_rewards_arr.append(shaped_reward)

            ep_reward += env_reward

            # === Second pass: learning=True, apply TD error ===
            actor.connections[('H', 'Y')].nu = lr_scale(running_reward, lr_out)

            if ti > 1 and saved_actor_state is not None:
                restore_actor_state(actor, saved_actor_state)
                actor.run(inputs={'X': spk_X}, time=time, reward=delta_t,
                          one_step=False, step=ti - 1, learning=True)

            state = next_state
            if terminated:
                break

        ep_length = ti
        running_reward = 0.05 * ep_reward + 0.95 * running_reward
        rewards_arr.append(ep_reward)
        ep_lengths_arr.append(ep_length)
        actor.reset_state_variables()

        # Best checkpoint by minimum episode length
        if ep_length < best_ep_length:
            best_ep_length = ep_length
            _save_checkpoint("best")
            print(f"  -> new best ep_length={ep_length} at ep {ep+1}, saved actor_best.pt")

        if (ep + 1) % 50 == 0:
            elapsed = now() - t0
            print(f"[ep {ep+1:5d}/{n_iter}]  "
                  f"ep_reward={ep_reward:7.1f}  "
                  f"ep_length={ep_length:5d}  "
                  f"running={running_reward:7.1f}  "
                  f"elapsed={elapsed/60:.1f} min")

        if (ep + 1) % save_every == 0:
            _save_checkpoint(f"ep{ep+1}")
            print(f"  -> saved checkpoint ep{ep+1}")

    env.close()

    _save_checkpoint("final")
    print(f"\nTraining done. Saved to {ACTOR_SAVE_DIR}")
    print(f"Total time: {(now()-t0)/60:.1f} min")
    print(f"Best episode length: {best_ep_length}")

    return rewards_arr, ep_lengths_arr


if __name__ == "__main__":
    seed = 0
    n_iter = 7000

    n_neurons_hid = 20    # must match pretrained hidden layer
    n_neurons_out = 3     # one per Acrobot action

    inhib_hid = 0.1
    inhib_out = 0.1

    lr_hid = 1e-5         # H layer is frozen (NoOp), this is unused
    lr_out = 1e-4         # H→Y MSTDPET learning rate

    time_ = 20            # simulation window (ms)
    dt = 0.5

    theta_plus = 0.0
    tc_theta_decay = 1e7

    w_init_hid = 0.7      # not used (H weights loaded from checkpoint)
    w_init_out = 0.5

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

    gamma = 0.99
    lda = 0.5       # eligibility trace decay
    alpha = 0.001   # critic learning rate

    epsilon = 0.0

    centers_per_feature = (12, 12, 12, 12, 10, 10)
    sigma_scale = 0.6
    rho = 500.0
    normalize_per_feature = True

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
        gamma=gamma,
        lda=lda,
        alpha=alpha,
        epsilon=epsilon,
        centers_per_feature=centers_per_feature,
        sigma_scale=sigma_scale,
        rho=rho,
        normalize_per_feature=normalize_per_feature,
        save_every=50,
    )

    print("ok")
