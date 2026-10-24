"""
Обучение выходного слоя актора (H→Y, MSTDPET) в среде CartPole-v0.
Скрытый слой (X→H, 500 нейронов, hidden_layers/cartpole_500n) заморожен.
Кодирование актора: 4 фичи × 15 центров, rho=5000.
Критик — нелинейный (Fourier basis), неспайковый, обучается TD(λ).

Сохранение:
  - лучший чекпойнт (по наивысшему ep_reward): best_net.pt + best_arrays.pt
  - периодически раз в checkpoint_interval эпизодов: ckpt_ep{N}_net.pt + ckpt_ep{N}_arrays.pt
"""

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, REPO_ROOT)  # local modified bindsnet

HIDDEN_LAYER_PATH = os.path.join(REPO_ROOT, "hidden_layers", "cartpole_500n", "hidden_layer.pt")

import json
import math
import random
import itertools
import warnings
from time import time as now
from typing import List, Optional

import numpy as np
import torch
import gymnasium as gym

from bindsnet.encoding import poisson
from bindsnet.network.monitors import Monitor
from bindsnet.models import Diehl2LayersWTAOutput

warnings.filterwarnings("ignore", category=UserWarning)


# =========================================================
# ENCODER
# =========================================================
class InputLayer:
    _RANGES = [
        (-2.5,  2.5),
        (-1.6,  1.6),
        (-0.45, 0.45),
        (-2.6,  2.6),
    ]

    def __init__(self, num_points: int = 15, rho: float = 5000.0):
        self.num_features = 4
        self.num_points   = num_points
        self.rho          = rho
        h = np.zeros(self.num_features)
        all_values = []
        for i, (lo, hi) in enumerate(self._RANGES):
            vals = np.linspace(lo, hi, num_points, dtype=np.float32)
            h[i] = vals[1] - vals[0]
            all_values.append(vals)
        self.h      = h.reshape(self.num_features, 1)
        self.values = np.vstack(all_values)
        self.n_inpt = self.num_features * self.num_points

    def step(self, state) -> torch.Tensor:
        s    = np.array(state, dtype=np.float32).reshape(self.num_features, 1)
        exp_ = np.exp(-0.5 * (self.values - s) ** 2 / self.h ** 2)
        return self.rho * torch.from_numpy(exp_.flatten()).float()


# =========================================================
# FOURIER BASIS CRITIC
# =========================================================
def make_fourier_combns(order: int = 5, n_dims: int = 4) -> np.ndarray:
    return np.array(list(itertools.product(range(order + 1), repeat=n_dims)))


def fourier_features(
    state: np.ndarray,
    combns: np.ndarray,
    lims: List[tuple],
) -> np.ndarray:
    s = np.zeros(len(lims))
    for i, (lo, hi) in enumerate(lims):
        s[i] = (state[i] - lo) / (hi - lo)
    return np.cos(np.pi * combns.dot(s))


# =========================================================
# ACTOR STATE SAVE / RESTORE
# =========================================================
def store_actor_state(net) -> dict:
    state = {
        "X": {"x": net.layers["X"].x.clone().detach()},
        "H": {
            "v":            net.layers["H"].v.clone().detach(),
            "x":            net.layers["H"].x.clone().detach(),
            "s":            net.layers["H"].s.clone().detach(),
            "refrac_count": net.layers["H"].refrac_count.clone().detach(),
            "decay":        net.layers["H"].decay.clone().detach(),
        },
        "Y": {
            "v":            net.layers["Y"].v.clone().detach(),
            "x":            net.layers["Y"].x.clone().detach(),
            "s":            net.layers["Y"].s.clone().detach(),
            "refrac_count": net.layers["Y"].refrac_count.clone().detach(),
            "decay":        net.layers["Y"].decay.clone().detach(),
        },
        "connections": {
            "H_Y": {
                "p_plus":            None,
                "p_minus":           None,
                "eligibility":       None,
                "eligibility_trace": None,
            }
        },
    }
    rule = net.connections[("H", "Y")].update_rule
    if hasattr(rule, "p_plus") and rule.p_plus is not None:
        state["connections"]["H_Y"]["p_plus"] = rule.p_plus.clone().detach()
    if hasattr(rule, "p_minus") and rule.p_minus is not None:
        state["connections"]["H_Y"]["p_minus"] = rule.p_minus.clone().detach()
    if hasattr(rule, "eligibility") and rule.eligibility is not None:
        state["connections"]["H_Y"]["eligibility"] = rule.eligibility.clone().detach()
    if hasattr(rule, "eligibility_trace") and rule.eligibility_trace is not None:
        state["connections"]["H_Y"]["eligibility_trace"] = rule.eligibility_trace.clone().detach()
    return state


def restore_actor_state(net, state: dict):
    net.layers["X"].x.copy_(state["X"]["x"])
    net.layers["H"].v.copy_(state["H"]["v"])
    net.layers["H"].x.copy_(state["H"]["x"])
    net.layers["H"].s.copy_(state["H"]["s"])
    net.layers["H"].refrac_count.copy_(state["H"]["refrac_count"])
    net.layers["H"].decay.copy_(state["H"]["decay"])
    net.layers["Y"].v.copy_(state["Y"]["v"])
    net.layers["Y"].x.copy_(state["Y"]["x"])
    net.layers["Y"].s.copy_(state["Y"]["s"])
    net.layers["Y"].refrac_count.copy_(state["Y"]["refrac_count"])
    net.layers["Y"].decay.copy_(state["Y"]["decay"])
    rule = net.connections[("H", "Y")].update_rule
    if state["connections"]["H_Y"]["p_plus"] is not None:
        rule.p_plus.copy_(state["connections"]["H_Y"]["p_plus"])
    if state["connections"]["H_Y"]["p_minus"] is not None:
        rule.p_minus.copy_(state["connections"]["H_Y"]["p_minus"])
    if state["connections"]["H_Y"]["eligibility"] is not None:
        rule.eligibility.copy_(state["connections"]["H_Y"]["eligibility"])
    if state["connections"]["H_Y"]["eligibility_trace"] is not None:
        rule.eligibility_trace.copy_(state["connections"]["H_Y"]["eligibility_trace"])


# =========================================================
# UTILS
# =========================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_dirs(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)


def safe_reset(net):
    for layer in net.layers.values():
        layer.reset_state_variables()
    for conn in net.connections.values():
        rule = conn.update_rule
        if hasattr(rule, "reset_state_variables"):
            rule.reset_state_variables()
    for monitor in net.monitors.values():
        monitor.reset_state_variables()



def e_greedy_action(action_ind: int, epsilon: float = 0.1) -> int:
    prob = np.full(2, epsilon / 2)
    prob[action_ind] = 1.0 - epsilon + epsilon / 2
    arr = np.concatenate([
        np.ones(int(100 * prob[1])),
        np.zeros(int(100 * prob[0])),
    ])
    return int(arr[random.randint(0, len(arr) - 1)])


# =========================================================
# CHECKPOINT SAVE
# =========================================================
def save_checkpoint(
    path_net: str,
    path_arrays: str,
    net,
    w_v: np.ndarray,
    rewards_arr: list,
    values_arr: list,
    td_errors_arr: list,
):
    net.save(path_net)
    torch.save(
        {
            "w_v":           torch.from_numpy(w_v),
            "rewards_arr":   rewards_arr,
            "values_arr":    values_arr,
            "td_errors_arr": td_errors_arr,
        },
        path_arrays,
    )


# =========================================================
# MAIN TRAINING
# =========================================================
def train(
    # --- пути ---
    hidden_layer_path: str = HIDDEN_LAYER_PATH,
    root_dir: str = SCRIPT_DIR,
    run_name: Optional[str] = None,

    # --- архитектура скрытого слоя (должна совпадать с чекпойнтом!) ---
    n_neurons_hid: int    = 500,
    inhib_hid: float      = 0.1,
    thresh_hid: float     = -58.0,
    refrac_hid: int       = 5,
    one_spike_hid: bool   = False,
    tc_decay_hid: float   = 15.0,
    tc_trace_hid: float   = 10.0,

    # --- архитектура выходного слоя ---
    n_neurons_out: int    = 2,
    inhib_out: float      = 0.1,
    lr_out: float         = 1e-3,
    w_init_out: float     = 0.5,
    thresh_out: float     = -62.0,
    refrac_out: int       = 0,
    one_spike_out: bool   = True,
    tc_decay_out: float   = 15.0,
    tc_trace_out: float   = 10.0,
    tc_plus_out: float    = 10.0,
    tc_minus_out: float   = 10.0,
    tc_e_trace_out: float = 15.0,
    wmin_out: float       = 0.0,
    wmax_out: float       = 1.0,

    # --- энкодер ---
    num_points: int       = 15,
    rho: float            = 5000.0,

    # --- симуляция ---
    time: int             = 20,
    dt: float             = 0.5,
    tc_trace_input: float = 20.0,

    # --- RL ---
    seed: int             = 0,
    n_iter: int           = 7000,
    gamma: float          = 0.99,
    epsilon: float        = 0.1,
    alpha_critic: float   = 0.001,
    lambda_critic: float  = 0.5,
    fourier_order: int    = 5,

    # --- чекпойнты ---
    checkpoint_interval: int = 100,
):
    set_seed(seed)

    if run_name is None:
        run_name = (
            f"out_hid{n_neurons_hid}_out{n_neurons_out}"
            f"_lr{lr_out}_thr{thresh_out}_inh{inhib_out}"
            f"_eps{epsilon}_g{gamma}_cartpole_v0"
        )

    run_dir = os.path.join(root_dir, "runs", run_name)
    ensure_dirs(run_dir)

    print("=" * 65)
    print(f"RUN: {run_name}")
    print(f"  hidden_layer: "
          f"{os.path.basename(os.path.dirname(hidden_layer_path))}/best.pt")
    print(f"  n_hid={n_neurons_hid}  n_out={n_neurons_out}")
    print(f"  lr_out={lr_out}  thresh_out={thresh_out}  inhib_out={inhib_out}")
    print(f"  epsilon={epsilon}  gamma={gamma}")
    print(f"  n_iter={n_iter}  seed={seed}")
    print(f"  run_dir={run_dir}")
    print("=" * 65)

    # ---- Сеть ----
    actor = Diehl2LayersWTAOutput(
        n_inpt=num_points * 4,
        n_neurons_hid=n_neurons_hid,
        n_neurons_out=n_neurons_out,
        inh_hid=inhib_hid,
        inh_out=inhib_out,
        dt=dt,
        nu_hid=[0.0, 0.0],
        nu_out=[0.4 * lr_out, lr_out],
        reduction=None,
        wmin=wmin_out,
        wmax=wmax_out,
        norm=None,
        theta_plus=0.0,
        tc_theta_decay=1e7,
        inpt_shape=None,
        hidden_layer_path=hidden_layer_path,
        w_init_hid=0.5,
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

    timesteps = int(round(time / dt))

    spikes = {}
    for layer_name in actor.layers:
        mon = Monitor(actor.layers[layer_name], state_vars=["s"], time=timesteps)
        actor.add_monitor(mon, name=f"{layer_name}_spikes")
        spikes[layer_name] = mon

    # ---- Энкодер ----
    encoder = InputLayer(num_points=num_points, rho=rho)
    print(f"Encoder: {num_points} pts/feature → {encoder.n_inpt} input neurons\n")

    # ---- Fourier-критик ----
    critic_lims = [(-3.0, 3.0), (-10.0, 10.0),
                   (-math.pi / 2, math.pi / 2), (-math.pi, math.pi)]
    combns = make_fourier_combns(order=fourier_order, n_dims=4)
    n_feat = len(combns)
    w_v    = np.zeros(n_feat, dtype=np.float64)

    # ---- Среда ----
    env = gym.make("CartPole-v0", render_mode="human")

    rewards_arr:   List[float] = []
    values_arr:    List[float] = []
    td_errors_arr: List[float] = []
    running_reward = 10.0
    best_ep_reward = -np.inf
    global_start   = now()

    # ---- Сохраняем конфиг ----
    cfg = dict(
        run_name=run_name, hidden_layer_path=hidden_layer_path,
        n_neurons_hid=n_neurons_hid, n_neurons_out=n_neurons_out,
        inhib_hid=inhib_hid, inhib_out=inhib_out,
        lr_out=lr_out, thresh_hid=thresh_hid, thresh_out=thresh_out,
        refrac_hid=refrac_hid, refrac_out=refrac_out,
        one_spike_hid=one_spike_hid, one_spike_out=one_spike_out,
        num_points=num_points, rho=rho, time=time, dt=dt,
        seed=seed, n_iter=n_iter, gamma=gamma,
        epsilon=epsilon, alpha_critic=alpha_critic,
        lambda_critic=lambda_critic, fourier_order=fourier_order,
    )
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    # ================================================================
    # ОСНОВНОЙ ЦИКЛ
    # ================================================================
    for ep in range(n_iter):
        raw_state = env.reset(seed=seed + ep)
        state     = raw_state[0] if isinstance(raw_state, tuple) else raw_state
        ep_reward = 0.0
        e_v       = np.zeros(n_feat, dtype=np.float64)

        safe_reset(actor)

        for step in range(1, 10001):
            rates_enc = encoder.step(state)
            sample    = poisson(datum=rates_enc, time=time, dt=dt)
            inpts     = {"X": sample}

            phi_s = fourier_features(state, combns, critic_lims)

            if step > 1:
                actor_state_buf = store_actor_state(actor)

            actor.run(inputs=inpts, time=time, reward=0.0,
                      one_step=False, step=step - 1, learning=False)

            spike_counts = spikes["Y"].get("s").sum(dim=0)[0]
            pred         = (spike_counts[1] >= spike_counts[0]).long()
            action       = e_greedy_action(pred, epsilon)

            next_state, reward, terminated, truncated, _ = env.step(action)
            reward = float(reward)
            done   = terminated or truncated
            if done:
                reward = 0.0

            phi_ns  = fourier_features(next_state, combns, critic_lims)
            v_s     = float(w_v.dot(phi_s))
            v_ns    = float(w_v.dot(phi_ns))
            if done:
                v_ns = 0.0
            delta_t = reward + gamma * v_ns - v_s

            e_v  = gamma * lambda_critic * e_v + phi_s
            w_v += alpha_critic * delta_t * e_v

            ep_reward += reward

            if step > 1:
                restore_actor_state(actor, actor_state_buf)
                actor.run(inputs=inpts, time=time, reward=delta_t,
                          one_step=False, step=step - 1, learning=True)

            td_errors_arr.append(delta_t)
            values_arr.append(v_ns)

            state = next_state

            if done:
                break

        running_reward = 0.05 * ep_reward + 0.95 * running_reward
        rewards_arr.append(ep_reward)

        print(
            f"[ep {ep+1:5d}]  reward={ep_reward:6.1f}  "
            f"running={running_reward:7.2f}  steps={step:4d}",
            flush=True,
        )

        if ep_reward >= 185:
            best_ep_reward = ep_reward
            save_checkpoint(
                os.path.join(run_dir, f"prelim1_net_ep_{ep}.pt"),
                os.path.join(run_dir, f"prelim_arrays_ep_{ep}.pt"),
                actor, w_v, rewards_arr, values_arr, td_errors_arr,
            )
            print(f"  >> Prelim ep_reward={ep_reward:.1f}")
        # --- Лучший чекпойнт ---
        if ep_reward > best_ep_reward:
            best_ep_reward = ep_reward
            save_checkpoint(
                os.path.join(run_dir, "best_net.pt"),
                os.path.join(run_dir, "best_arrays.pt"),
                actor, w_v, rewards_arr, values_arr, td_errors_arr,
            )
            print(f"  >> Новый лучший  ep_reward={ep_reward:.1f}")

        # --- Периодический чекпойнт ---
        if (ep + 1) % checkpoint_interval == 0:
            ep_tag = f"{ep+1:06d}"
            save_checkpoint(
                os.path.join(run_dir, f"ckpt_ep{ep_tag}_net.pt"),
                os.path.join(run_dir, f"ckpt_ep{ep_tag}_arrays.pt"),
                actor, w_v, rewards_arr, values_arr, td_errors_arr,
            )

    env.close()
    total_time = now() - global_start
    print(f"\nОбучение завершено за {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Лучший ep_reward: {best_ep_reward:.1f}")
    print(f"Файлы в: {run_dir}")


# =========================================================
# ЗАПУСК
# =========================================================
if __name__ == "__main__":
    train(
        hidden_layer_path=HIDDEN_LAYER_PATH,
        root_dir=SCRIPT_DIR,
        run_name=None,

        # --- скрытый слой (должен совпадать с чекпойнтом!) ---
        n_neurons_hid=500,
        inhib_hid=0.1,
        thresh_hid=-58.0,
        refrac_hid=5,
        one_spike_hid=False,
        tc_decay_hid=15.0,
        tc_trace_hid=10.0,

        # --- выходной слой ---
        n_neurons_out=2,
        inhib_out=0.1,
        lr_out=2e-4,
        w_init_out=0.5,
        thresh_out=-61.0,
        refrac_out=0,
        one_spike_out=True,
        tc_decay_out=20.0,
        tc_trace_out=20.0,
        tc_plus_out=20.0,
        tc_minus_out=20.0,
        tc_e_trace_out=20.0,
        wmin_out=0.0,
        wmax_out=1.0,

        # --- энкодер ---
        num_points=15,
        rho=5000.0,

        # --- симуляция ---
        time=20,
        dt=0.5,
        tc_trace_input=20.0,

        # --- RL ---
        seed=0,
        n_iter=200_000,
        gamma=0.99,
        epsilon=0.0,
        alpha_critic=0.001,
        lambda_critic=0.5,
        fourier_order=5,

        # --- сохранение ---
        checkpoint_interval=100,
    )
