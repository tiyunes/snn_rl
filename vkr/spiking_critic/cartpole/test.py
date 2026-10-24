"""
Greedy evaluation of an actor checkpoint produced by train.py.

The checkpoint format from save_checkpoint() in train.py is a
plain dict (not a bindsnet-pickled Network), so we:
  1. Rebuild the actor with the same architecture (Diehl2LayersWTAOutput,
     500 hidden neurons, 2-neuron output, 60-input GRF encoding).
  2. Copy weights and theta from the checkpoint into the rebuilt actor.
  3. Run N CartPole-v0 episodes with action = argmax(Y spike counts)
     and learning disabled.
  4. Save rewards/lengths and a quick plot.

The critic is irrelevant for testing the policy and is not loaded.
"""

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, REPO_ROOT)  # local modified bindsnet

import glob
import random
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import torch
import gymnasium as gym
import matplotlib.pyplot as plt

from bindsnet.encoding import poisson
from bindsnet.learning import NoOp
from bindsnet.network.monitors import Monitor
from bindsnet.models import Diehl2LayersWTAOutput


# ---------------------------
# PATHS
# ---------------------------
HIDDEN_LAYER_PATH = os.path.join(REPO_ROOT, "hidden_layers", "cartpole_500n", "hidden_layer.pt")

# Directory with checkpoints written by train.py (used when ckpt_path is None).
RUN_DIR = os.path.join(SCRIPT_DIR, "runs")
CKPT_PATH = os.path.join(SCRIPT_DIR, "checkpoint.pt")
# ---------------------------
# COMMON HELPERS (same as in train.py)
# ---------------------------
def set_seeds(seed: int = 0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def reset_activity_only(net):
    """Zero v / s / x / refrac_count and reset monitors. Preserve theta."""
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


class InputLayer:
    """4-feature GRF encoder, same as in the training script (60 input neurons)."""

    def __init__(self, rho: float = 5000.0, num_points: int = 15):
        self.num_features = 4
        self.rho = float(rho)
        self.num_points = int(num_points)

        self.h = np.zeros(self.num_features)

        x_values = np.linspace(-2.5, 2.5, num_points)
        self.h[0] = x_values[1] - x_values[0]

        vel_values = np.linspace(-1.6, 1.6, num_points)
        self.h[1] = vel_values[1] - vel_values[0]

        theta_values = np.linspace(-0.45, 0.45, num_points)
        self.h[2] = theta_values[1] - theta_values[0]

        theta_vel_values = np.linspace(-2.6, 2.6, num_points)
        self.h[3] = theta_vel_values[1] - theta_vel_values[0]

        self.h = self.h.reshape(self.num_features, 1)
        self.values = np.vstack((x_values, vel_values, theta_values, theta_vel_values))
        self.n_inpt = self.num_features * self.num_points

    def rates(self, state) -> torch.Tensor:
        s = np.asarray(state, dtype=np.float32).reshape(self.num_features, 1)
        diff = (self.values - s) ** 2
        exp = np.exp(-0.5 * diff / (self.h ** 2))
        return torch.from_numpy((self.rho * exp.flatten()).astype(np.float32))


def build_grf_spike_train(encoder: InputLayer, state, sim_time: float, dt: float) -> torch.Tensor:
    rates = encoder.rates(np.asarray(state, dtype=np.float32))
    spk = poisson(datum=rates, time=sim_time, dt=dt)
    if spk.ndim == 2:
        spk = spk.unsqueeze(1)
    return spk.float()


# ---------------------------
# ACTOR BUILD + LOAD
# ---------------------------
def build_actor_net(
    seed: int = 0,
    dt: float = 0.5,
    n_inpt: int = 60,
    n_actions: int = 2,
    n_neurons_hid: int = 500,
    inhib_hid: float = 0.1,
    inhib_out: float = 0.1,
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
    """Build the actor architecture used in training. nu is zeroed and every
    connection's update rule is replaced with NoOp — we only run forward."""
    set_seeds(seed)

    net = Diehl2LayersWTAOutput(
        n_inpt=n_inpt,
        n_neurons_hid=n_neurons_hid,
        n_neurons_out=n_actions,
        inh_hid=inhib_hid,
        inh_out=inhib_out,
        dt=dt,
        nu_hid=[0.0, 0.0],
        nu_out=[0.0, 0.0],
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

    for conn in net.connections.values():
        conn.update_rule = NoOp(connection=conn, nu=getattr(conn, "nu", None))

    return net


def find_best_checkpoint(run_dir: str) -> str:
    """Pick the latest entry from best_checkpoints.csv if present, else fall
    back to the highest-ep `online_spike_critic_best_ep*_lr_1e_4.pt` file."""
    csv_path = os.path.join(run_dir, "best_checkpoints.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        if len(df) > 0:
            return str(df["checkpoint_path"].iloc[-1])

    pattern = os.path.join(run_dir, "online_spike_critic_best_ep*_lr_1e_4.pt")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(
            f"No 'online_spike_critic_best_ep*' checkpoints in {run_dir}"
        )

    def ep_num(p: str) -> int:
        # parse "...best_ep{N}_lr_1e_4.pt"
        name = os.path.basename(p)
        return int(name.split("ep")[1].split("_")[0])

    return sorted(files, key=ep_num)[-1]


def load_actor_from_checkpoint(net, ckpt_path: str) -> dict:
    """Copy weights and theta from a save_checkpoint() dict into `net`."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    n_loaded_conn, n_loaded_theta = 0, 0
    for key, conn in net.connections.items():
        key_str = str(key)
        if key_str in ckpt.get("actor_connections", {}):
            w_saved = ckpt["actor_connections"][key_str]["w"]
            conn.w.data.copy_(w_saved.to(conn.w.device))
            n_loaded_conn += 1

    for lname, layer in net.layers.items():
        layer_pack = ckpt.get("actor_layers", {}).get(lname, {})
        theta_t = layer_pack.get("theta") if isinstance(layer_pack, dict) else None
        if theta_t is not None and hasattr(layer, "theta") and torch.is_tensor(layer.theta):
            layer.theta.data.copy_(theta_t.to(layer.theta.device))
            n_loaded_theta += 1

    print(f"[LOAD] {n_loaded_conn} connections, {n_loaded_theta} theta tensors")
    return ckpt


# ---------------------------
# EVAL LOOP
# ---------------------------
@torch.no_grad()
def evaluate(
    actor_net,
    env,
    encoder: InputLayer,
    sim_time: float,
    dt: float,
    n_episodes: int,
    max_steps: int,
    seed: int = 0,
) -> Tuple[List[float], List[int]]:
    rewards: List[float] = []
    lengths: List[int] = []

    monitors = actor_net.monitors

    for ep in range(n_episodes):
        state, _ = env.reset(seed=seed + ep)
        reset_activity_only(actor_net)
        ep_reward = 0.0
        ti = 0

        for ti in range(1, max_steps + 1):
            spk_X = build_grf_spike_train(encoder, state, sim_time, dt)
            reset_activity_only(actor_net)  # matches actor_reset_activity=True in training
            actor_net.run(
                inputs={"X": spk_X},
                time=sim_time,
                learning=False,
                one_step=False,
            )
            y_spikes = monitors["Y_spikes"].get("s")
            y_counts = y_spikes.sum(dim=0)[0]
            action = int(torch.argmax(y_counts).item())

            state, reward, terminated, truncated, _ = env.step(action)
            ep_reward += float(reward)
            if terminated or truncated:
                break

        rewards.append(ep_reward)
        lengths.append(ti)
        print(f"  ep={ep + 1:4d} | reward={ep_reward:6.1f} | len={ti:4d}")

    return rewards, lengths


# ---------------------------
# MAIN
# ---------------------------
def main(
    run_dir: str = RUN_DIR,
    ckpt_path: Optional[str] = None,  # if None: auto-pick best from run_dir
    n_episodes: int = 100,
    max_steps: int = 200,
    sim_time: float = 20.0,
    dt: float = 0.5,
    seed: int = 0,
    render: bool = False,

    # actor architecture — must match training run
    n_neurons_hid: int = 500,
    inhib_hid: float = 0.1,
    inhib_out: float = 0.1,
    thresh_hid: float = -58.0,
    thresh_out: float = -64.0,
    refrac_hid: int = 5,
    refrac_out: int = 0,
    one_spike_hid: bool = False,
    one_spike_out: bool = True,
    actor_tc_trace_input: float = 20.0,
    actor_tc_decay_hid: float = 15.0,
    actor_tc_decay_out: float = 15.0,
    actor_tc_trace_hid: float = 10.0,
    actor_tc_trace_out: float = 10.0,
    actor_tc_plus_out: float = 10.0,
    actor_tc_minus_out: float = 10.0,
    actor_tc_e_trace_out: float = 15.0,
    rho: float = 5000.0,

    hidden_layer_path: str = HIDDEN_LAYER_PATH,
    save_npz: bool = True,
    save_plot: bool = True,
):
    set_seeds(seed)

    if ckpt_path is None:
        ckpt_path = find_best_checkpoint(run_dir)
    print(f"[EVAL] Checkpoint: {ckpt_path}")

    encoder = InputLayer(rho=rho)
    timesteps = int(round(sim_time / dt))

    actor = build_actor_net(
        seed=seed,
        dt=dt,
        n_inpt=encoder.n_inpt,
        n_actions=2,
        n_neurons_hid=n_neurons_hid,
        inhib_hid=inhib_hid,
        inhib_out=inhib_out,
        thresh_hid=thresh_hid,
        thresh_out=thresh_out,
        refrac_hid=refrac_hid,
        refrac_out=refrac_out,
        one_spike_hid=one_spike_hid,
        one_spike_out=one_spike_out,
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

    ckpt = load_actor_from_checkpoint(actor, ckpt_path)
    extra = ckpt.get("extra", {})
    print(
        f"[EVAL] Source ep={extra.get('episode', '?')}, "
        f"training best_mean_len={extra.get('best_mean_len', '?')}"
    )

    actor.add_monitor(
        Monitor(actor.layers["Y"], state_vars=["s"], time=timesteps),
        name="Y_spikes",
    )

    env = gym.make("CartPole-v0", render_mode="human" if render else None)

    rewards, lengths = evaluate(
        actor_net=actor,
        env=env,
        encoder=encoder,
        sim_time=sim_time,
        dt=dt,
        n_episodes=n_episodes,
        max_steps=max_steps,
        seed=seed,
    )
    env.close()

    rewards = np.asarray(rewards, dtype=np.float64)
    lengths = np.asarray(lengths, dtype=np.int64)

    print()
    print("=" * 60)
    print(f"[EVAL] reward  mean={rewards.mean():.2f}  std={rewards.std():.2f}  "
          f"min={rewards.min():.0f}  max={rewards.max():.0f}")
    print(f"[EVAL] length  mean={lengths.mean():.2f}  std={lengths.std():.2f}  "
          f"max={int(lengths.max())}")
    print(f"[EVAL] success(@{max_steps}) = "
          f"{(lengths >= max_steps).mean():.2%}  "
          f"({int((lengths >= max_steps).sum())}/{n_episodes} eps)")
    print("=" * 60)

    os.makedirs(run_dir, exist_ok=True)
    if save_npz:
        out_npz = os.path.join(run_dir, "eval_best_rewards.npz")
        np.savez(
            out_npz,
            rewards=rewards,
            lengths=lengths,
            ckpt_path=ckpt_path,
            max_steps=max_steps,
        )
        print(f"[EVAL] Saved: {out_npz}")

    if save_plot:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        axes[0].plot(rewards, marker="o", linewidth=1, alpha=0.6)
        axes[0].axhline(max_steps, color="green", linestyle="--", linewidth=1,
                        label=f"max_steps={max_steps}")
        axes[0].set_xlabel("episode")
        axes[0].set_ylabel("reward")
        axes[0].set_title(f"Best checkpoint reward ({n_episodes} eps)")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].hist(rewards, bins=20, color="steelblue", alpha=0.8)
        axes[1].axvline(rewards.mean(), color="red", linestyle="--", linewidth=1,
                        label=f"mean={rewards.mean():.1f}")
        axes[1].set_xlabel("reward")
        axes[1].set_ylabel("count")
        axes[1].set_title("Reward distribution")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        out_png = os.path.join(run_dir, "eval_best_rewards.png")
        plt.savefig(out_png, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"[EVAL] Saved: {out_png}")

    return rewards, lengths


if __name__ == "__main__":
    main(
        run_dir=RUN_DIR,
        ckpt_path=CKPT_PATH,        # None -> auto-pick best from run_dir
        n_episodes=100,
        max_steps=200,
        sim_time=20.0,
        dt=0.5,
        seed=0,
        render=False,          # True to visually watch episodes (slower)

        # actor architecture — must match the training run
        n_neurons_hid=500,
        inhib_hid=0.1,
        inhib_out=0.1,
        thresh_hid=-58.0,
        thresh_out=-63.0,
        refrac_hid=5,
        refrac_out=0,
        one_spike_hid=False,
        one_spike_out=True,
        rho=5000.0,

        hidden_layer_path=HIDDEN_LAYER_PATH,
    )
    print("ok")
