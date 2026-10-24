# ===========================
# MountainCar SNN Actor + Linear Fourier Critic
# - Fixed pretrained hidden layer X->H
# - Trains only output spiking actor layer H->Y
# - Critic is linear Fourier TD(lambda)
# - Encoder matches hidden-layer pretraining (hidden_layers/mountain_car_500n)
# ===========================

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, REPO_ROOT)  # local modified bindsnet

import json
import random
import itertools
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import gymnasium as gym
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from bindsnet.encoding import poisson
from bindsnet.learning import NoOp
from bindsnet.network.monitors import Monitor
from bindsnet.models import Diehl2LayersWTAOutput


# ---------------------------
# PATHS
# ---------------------------
DATA_PATH = os.path.join(REPO_ROOT, "data", "mountaincar_state_action.csv")
HIDDEN_LAYER_PATH = os.path.join(REPO_ROOT, "hidden_layers", "mountain_car_500n", "hidden_layer.pt")
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


def get_scale(mean_reward, denom=200.0, min_scale=0.05, max_scale=1.0):
    """
    Dynamic scale for actor output-layer learning rate.
    For MountainCar returns are typically around [-200, ...].
    """
    scale = abs(float(mean_reward)) / float(denom)
    scale = max(min_scale, min(max_scale, scale))
    return float(scale)


def e_greedy_action(action_ind: int, epsilon=0.1, n_actions=3) -> int:
    probs = np.full(n_actions, epsilon / n_actions, dtype=np.float64)
    probs[action_ind] += (1.0 - epsilon)
    return int(np.random.choice(np.arange(n_actions), p=probs))


# ---------------------------
# DATA / ENCODER SUPPORT
# ---------------------------
def load_states_from_csv(csv_path: str) -> np.ndarray:
    df = pd.read_csv(csv_path)

    required_cols = ["position", "velocity"]
    missing = [c for c in required_cols if c not in df.columns]
    if len(missing) > 0:
        raise ValueError(f"Missing required columns in dataset: {missing}")

    states_np = df[["position", "velocity"]].to_numpy(dtype=np.float32)
    return states_np


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


class InputLayerMountainCar:
    """
    Matches hidden-layer pretraining:
    - compact GRF
    - 2 features: position, velocity
    - sigma = sigma_scale * spacing
    - optional per-feature normalization
    """

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

        assert len(self.centers_per_feature) == self.num_features

        if feature_ranges is None:
            feature_ranges = [
                (-1.2, 0.6),
                (-0.07, 0.07),
            ]

        self.feature_ranges = feature_ranges
        self.centers: List[np.ndarray] = []
        self.sigmas: List[np.ndarray] = []

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
        return rates.astype(np.float32)

    def rates(self, state):
        return torch.tensor(
            self.encode_numpy(state),
            dtype=torch.float32,
            device=self.device,
        )


# ---------------------------
# FOURIER CRITIC
# ---------------------------
def fourier_feature_state_mc(state, low, high, combns):
    state = np.asarray(state, dtype=np.float32)
    state_norm = (state - low) / (high - low + 1e-12)
    prod_array = np.array([np.dot(state_norm, c) for c in combns], dtype=np.float32)
    features = np.cos(np.pi * prod_array)
    return features.astype(np.float32)


# ---------------------------
# SNAPSHOT / RESET
# ---------------------------
def safe_reset(net):
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


# ---------------------------
# NETWORK BUILD
# ---------------------------
def build_actor_net(
    seed=0,
    dt=0.5,
    n_inpt=50,
    n_actions=3,
    n_neurons_hid=500,
    inhib_hid=0.1,
    inhib_out=0.1,
    lr_hid=1e-7,
    lr_out=5e-4,
    w_init_hid=1.0,
    w_init_out=0.5,
    thresh_hid=-58.0,
    thresh_out=-64.0,
    refrac_hid=0,
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

    # Freeze hidden layer learning
    if ("X", "H") in net.connections:
        net.connections[("X", "H")].update_rule = NoOp(
            connection=net.connections[("X", "H")],
            nu=net.connections[("X", "H")].nu,
        )

    # Actor reward sign setup
    conn_actor = net.connections[("H", "Y")]
    if hasattr(conn_actor.update_rule, "a_minus"):
        conn_actor.update_rule.a_minus = -1.0
    if hasattr(conn_actor.update_rule, "a_plus"):
        conn_actor.update_rule.a_plus = 1.0

    return net


def get_spike_counts(spikes: dict, layer_name: str) -> torch.Tensor:
    s = spikes[layer_name].get("s")
    return s.sum(dim=0)[0].detach().clone()


# ---------------------------
# CHECKPOINT SAVE
# ---------------------------
def save_checkpoint(
    save_dir,
    episode_idx,
    actor_network,
    critic_w,
    rewards_arr,
    td_errors_arr,
    values_arr,
    final_positions_arr,
    running_reward,
    meta: dict,
):
    ensure_dirs(save_dir)

    actor_path = os.path.join(save_dir, f"actor_ep_{episode_idx}.pt")
    critic_path = os.path.join(save_dir, f"critic_ep_{episode_idx}.npy")
    summary_path = os.path.join(save_dir, f"summary_ep_{episode_idx}.json")

    actor_network.save(actor_path)
    np.save(critic_path, critic_w)

    payload = {
        "episode_idx": int(episode_idx),
        "running_reward": float(running_reward),
        "last_reward": float(rewards_arr[-1]) if len(rewards_arr) else None,
        "mean_reward_100": summarize_last_window(rewards_arr, 100),
        "mean_len_100": summarize_last_window(meta["ep_len_arr"], 100),
        "success_rate_100": summarize_last_window(meta["success_arr"], 100),
        "mean_final_position_100": summarize_last_window(final_positions_arr, 100),
        "meta": meta,
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"[Checkpoint] saved at episode {episode_idx} -> {actor_path}")


# ---------------------------
# MAIN
# ---------------------------
def main(
    seed=0,
    n_episodes=4000,
    max_steps=200,
    time_sec=20.0,
    dt=0.5,
    gamma_linear=0.99,
    lambda_linear=0.5,
    alpha_linear=0.001,
    epsilon=0.1,
    rho=1500.0,
    centers_per_feature=(30, 20),
    sigma_scale=0.6,
    is_input_normalize=True,
    q_low=0.01,
    q_high=0.99,
    fourier_order_linear=5,
    summary_every_episodes=25,
    save_every_episodes=100,
    lr_out=5e-4,
    lr_out_scale=True,
    terminal_bonus=5.0,
    render_mode="human",
    device=None,
):
    ensure_dirs(SAVE_DIR)
    set_seeds(seed)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    print(f"[Device] {device}")

    timesteps = int(round(time_sec / dt))
    sim_time = time_sec

    # ---------------------------
    # Environment
    # ---------------------------
    env = gym.make("MountainCar-v0", render_mode=render_mode)
    env.reset(seed=seed)

    # ---------------------------
    # Actor encoder matched to hidden-layer training
    # ---------------------------
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

    # ---------------------------
    # Linear critic
    # ---------------------------
    low = np.array([feature_ranges[0][0], feature_ranges[1][0]], dtype=np.float32)
    high = np.array([feature_ranges[0][1], feature_ranges[1][1]], dtype=np.float32)

    num_states = 2
    combns = np.array(
        list(itertools.product(range(fourier_order_linear + 1), repeat=num_states)),
        dtype=np.float32,
    )
    n_fourier_linear = len(combns)
    w_v = np.zeros(n_fourier_linear, dtype=np.float32)

    # ---------------------------
    # Actor network
    # ---------------------------
    net = build_actor_net(
        seed=seed,
        dt=dt,
        n_inpt=actor_enc.n_inpt,
        n_actions=3,
        n_neurons_hid=500,
        inhib_hid=0.1,
        inhib_out=0.1,
        lr_hid=1e-7,
        lr_out=lr_out,
        w_init_hid=1.0,
        w_init_out=0.5,
        thresh_hid=-58.0,
        thresh_out=-64.0,
        refrac_hid=0,
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
    )
    net.to(device)

    if ("X", "H") in net.connections:
        xh = net.connections[("X", "H")].w
        if xh.shape[0] != actor_enc.n_inpt:
            raise ValueError(
                f"Input dim mismatch: actor encoder n_inpt={actor_enc.n_inpt}, "
                f"but loaded hidden layer expects {xh.shape[0]}"
            )

    spikes = {}
    for layer in net.layers:
        spikes[layer] = Monitor(
            net.layers[layer],
            state_vars=["s"],
            time=timesteps,
        )
        net.add_monitor(spikes[layer], name=f"{layer}_spikes")

    # ---------------------------
    # Logs
    # ---------------------------
    rewards_arr = []
    ep_len_arr = []
    success_arr = []
    final_positions_arr = []

    values_linear_arr = []
    td_errors_linear_arr = []

    running_reward = -200.0
    best_mean_final_position = -1e9
    best_mean_reward = -1e9

    # ---------------------------
    # Training loop
    # ---------------------------
    for ep in tqdm(range(1, n_episodes + 1), desc="Train"):
        safe_reset(net)
        state, _ = env.reset(seed=seed + ep)

        e_v = np.zeros(n_fourier_linear, dtype=np.float32)
        ep_reward = 0.0
        success_flag = False

        for ti in range(1, max_steps + 1):
            if isinstance(state, tuple):
                state = state[0]

            # -------- build actor spikes for s_t --------
            rates_X = actor_enc.rates(state)
            spk_X = poisson(datum=rates_X, time=int(time_sec), dt=dt)
            inpts_t = {"X": spk_X}

            snap0 = store_state(net)

            # -------- A-pass: choose action --------
            net.run(
                inputs=inpts_t,
                time=sim_time,
                learning=False,
                reward=0.0,
                step=ti - 1,
                one_step=False,
            )

            y_counts = get_spike_counts(spikes, "Y")
            action_pred = int(torch.argmax(y_counts).item())
            action = e_greedy_action(action_pred, epsilon=epsilon, n_actions=3)

            # -------- critic values --------
            phi_s = fourier_feature_state_mc(
                state=state,
                low=low,
                high=high,
                combns=combns,
            )
            V_s = float(np.dot(w_v, phi_s))

            next_state, env_reward, terminated, truncated, _ = env.step(int(action))
            done = bool(terminated or truncated)

            # reward shaping
            if terminated:
                reward_used = float(terminal_bonus)
                success_flag = True
            else:
                reward_used = -1.0

            phi_ns = fourier_feature_state_mc(
                state=next_state,
                low=low,
                high=high,
                combns=combns,
            )
            V_ns = float(np.dot(w_v, phi_ns))

            if done and not terminated:
                target_linear = reward_used
            elif terminated:
                target_linear = reward_used
            else:
                target_linear = reward_used + gamma_linear * V_ns

            delta_linear = float(target_linear - V_s)

            # -------- linear critic update TD(lambda) --------
            e_v = gamma_linear * lambda_linear * e_v + phi_s
            w_v += alpha_linear * delta_linear * e_v

            # -------- actor output learning-rate scaling --------
            if lr_out_scale and ("H", "Y") in net.connections:
                scale = get_scale(running_reward, denom=500.0, min_scale=0.05, max_scale=1.0)
                conn = net.connections[("H", "Y")]
                conn.nu = [0.25 * lr_out * scale, lr_out * scale]

            # -------- actor learning-pass on s_t using delta_linear --------
            restore_state(net, snap0)
            net.run(
                inputs=inpts_t,
                time=sim_time,
                learning=True,
                reward=delta_linear,
                step=ti - 1,
                one_step=False,
            )

            # -------- logs --------
            ep_reward += reward_used
            values_linear_arr.append(float(V_s))
            td_errors_linear_arr.append(float(delta_linear))

            state = next_state

            if terminated:
                break

        running_reward = 0.05 * ep_reward + (1.0 - 0.05) * running_reward

        rewards_arr.append(ep_reward)
        ep_len_arr.append(ti)
        success_arr.append(int(success_flag))
        final_positions_arr.append(float(state[0]))

        if ep % summary_every_episodes == 0:
            print(
                f"EP={ep} | len={ti} | ep_reward={ep_reward:.2f} | "
                f"mean len(200)={summarize_last_window(ep_len_arr, 200):.2f} | "
                f"mean reward(200)={summarize_last_window(rewards_arr, 200):.2f} | "
                f"success rate(200)={summarize_last_window(success_arr, 200):.3f} | "
                f"mean final pos(200)={summarize_last_window(final_positions_arr, 200):.4f} | "
                f"mean V_linear(200)={summarize_last_window(values_linear_arr, 200):.4f} | "
                f"mean delta_linear(200)={summarize_last_window(td_errors_linear_arr, 200):.4f}"
            )

        # checkpoint by mean final position / mean reward
        mean_final_position_100 = summarize_last_window(final_positions_arr, 100)
        mean_reward_100 = summarize_last_window(rewards_arr, 100)

        need_save = False
        if mean_final_position_100 > best_mean_final_position:
            best_mean_final_position = mean_final_position_100
            need_save = True

        if mean_reward_100 > best_mean_reward:
            best_mean_reward = mean_reward_100
            need_save = True

        if ep % save_every_episodes == 0:
            need_save = True

        if need_save:
            meta = {
                "seed": seed,
                "n_episodes": n_episodes,
                "max_steps": max_steps,
                "time_sec": time_sec,
                "dt": dt,
                "gamma_linear": gamma_linear,
                "lambda_linear": lambda_linear,
                "alpha_linear": alpha_linear,
                "epsilon": epsilon,
                "rho": rho,
                "centers_per_feature": list(centers_per_feature),
                "sigma_scale": sigma_scale,
                "is_input_normalize": bool(is_input_normalize),
                "q_low": q_low,
                "q_high": q_high,
                "fourier_order_linear": fourier_order_linear,
                "lr_out": lr_out,
                "lr_out_scale": bool(lr_out_scale),
                "terminal_bonus": terminal_bonus,
                "hidden_layer_path": HIDDEN_LAYER_PATH,
                "feature_ranges": [list(x) for x in feature_ranges],
                "actor_encoder_n_inpt": actor_enc.n_inpt,
                "ep_len_arr": ep_len_arr[-200:],
                "success_arr": success_arr[-200:],
            }

            save_checkpoint(
                save_dir=SAVE_DIR,
                episode_idx=ep,
                actor_network=net,
                critic_w=w_v,
                rewards_arr=rewards_arr,
                td_errors_arr=td_errors_linear_arr,
                values_arr=values_linear_arr,
                final_positions_arr=final_positions_arr,
                running_reward=running_reward,
                meta=meta,
            )

        safe_reset(net)

    env.close()

    # ---------------------------
    # Final plots
    # ---------------------------
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

    if len(values_linear_arr) > 0:
        plt.figure(figsize=(8, 4))
        plt.plot(values_linear_arr[-2000:])
        plt.title("V_linear last 2000 steps")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    if len(td_errors_linear_arr) > 0:
        plt.figure(figsize=(8, 4))
        plt.plot(td_errors_linear_arr[-2000:])
        plt.title("delta_linear last 2000 steps")
        plt.grid(True)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main(
        seed=0,
        n_episodes=4000,
        max_steps=1000,
        time_sec=20.0,
        dt=0.5,
        gamma_linear=0.99,
        lambda_linear=0.5,
        alpha_linear=0.001,
        epsilon=0.0,
        rho=1500.0,
        centers_per_feature=(30, 20),
        sigma_scale=0.6,
        is_input_normalize=True,
        q_low=0.01,
        q_high=0.99,
        fourier_order_linear=5,
        summary_every_episodes=25,
        save_every_episodes=100,
        lr_out=1e-4,
        lr_out_scale=False,
        terminal_bonus=5.0,
        render_mode="human",   # set None for faster training
        device=None,
    )