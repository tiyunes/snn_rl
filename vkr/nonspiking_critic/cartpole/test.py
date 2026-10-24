"""
Жадная (greedy) оценка актора, обученного train.py.

  - Актор сохранён через `net.save()` (т.е. torch.save целиком),
    реконструировать архитектуру и копировать веса не нужно.
  - Действие выбирается так же, как при обучении:
        action = (spike_counts[1] >= spike_counts[0]).long()
    (при равенстве спайков выбираем класс 1 — как в e_greedy_action при ε=0).
  - Активность сети сбрасывается ТОЛЬКО в начале эпизода через safe_reset,
    между шагами эпизода активность сохраняется (как в обучении).
  - Энкодер 4 фичи × 15 центров, rho=5000 (как в train.py).
  - Среда — CartPole-v0.

"""

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, REPO_ROOT)  # local modified bindsnet

import random
from typing import List, Tuple

import numpy as np
import torch
import gymnasium as gym
import matplotlib.pyplot as plt

from bindsnet.encoding import poisson
from bindsnet.network.monitors import Monitor


# ---------------------------
# PATHS
# ---------------------------
CKPT_PATH = os.path.join(SCRIPT_DIR, "actor.pt")


# ---------------------------
# ENCODER (как в train.py)
# ---------------------------
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


# ---------------------------
# UTILS (safe_reset как в train.py)
# ---------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def safe_reset(net):
    for layer in net.layers.values():
        layer.reset_state_variables()
    for conn in net.connections.values():
        rule = conn.update_rule
        if hasattr(rule, "reset_state_variables"):
            rule.reset_state_variables()
    for monitor in net.monitors.values():
        monitor.reset_state_variables()


def load_actor(ckpt_path: str):
    """Загружаем pickled Network, сохранённый в обучении через actor.save()."""
    actor = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return actor


# ---------------------------
# EVAL LOOP
# ---------------------------
@torch.no_grad()
def evaluate(
    actor,
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

    y_mon = actor.monitors["Y_spikes"]

    for ep in range(n_episodes):
        raw_state = env.reset(seed=seed + ep)
        state = raw_state[0] if isinstance(raw_state, tuple) else raw_state
        safe_reset(actor)
        ep_reward = 0.0
        step = 0

        for step in range(1, max_steps + 1):
            rates_enc = encoder.step(state)
            sample    = poisson(datum=rates_enc, time=sim_time, dt=dt)

            actor.run(
                inputs={"X": sample},
                time=sim_time,
                reward=0.0,
                one_step=False,
                step=step - 1,
                learning=False,
            )

            spike_counts = y_mon.get("s").sum(dim=0)[0]
            action = int((spike_counts[1] >= spike_counts[0]).long().item())

            next_state, reward, terminated, truncated, _ = env.step(action)
            ep_reward += float(reward)
            state = next_state
            if terminated or truncated:
                break

        rewards.append(ep_reward)
        lengths.append(step)
        print(f"  ep={ep + 1:4d} | reward={ep_reward:6.1f} | len={step:4d}")

    return rewards, lengths


# ---------------------------
# MAIN
# ---------------------------
def main(
    ckpt_path: str = CKPT_PATH,
    n_episodes: int = 100,
    max_steps: int = 200,
    sim_time: float = 20.0,
    dt: float = 0.5,
    num_points: int = 15,
    rho: float = 5000.0,
    seed: int = 0,
    render: bool = False,
    save_npz: bool = True,
    save_plot: bool = True,
):
    set_seed(seed)

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Чекпойнт не найден: {ckpt_path}")
    print(f"[EVAL] Checkpoint: {ckpt_path}")

    encoder = InputLayer(num_points=num_points, rho=rho)
    timesteps = int(round(sim_time / dt))

    actor = load_actor(ckpt_path)

    # На случай если в сохранённой сети не оказалось монитора Y_spikes
    # (например, ckpt сохранён до его создания) — добавим.
    if "Y_spikes" not in actor.monitors:
        actor.add_monitor(
            Monitor(actor.layers["Y"], state_vars=["s"], time=timesteps),
            name="Y_spikes",
        )

    env = gym.make("CartPole-v0", render_mode="human" if render else None)

    rewards, lengths = evaluate(
        actor=actor,
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

    out_dir = os.path.dirname(os.path.abspath(ckpt_path))
    base    = os.path.splitext(os.path.basename(ckpt_path))[0]

    if save_npz:
        out_npz = os.path.join(out_dir, f"eval_{base}.npz")
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
        axes[0].set_title(f"Greedy reward ({n_episodes} eps)")
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
        out_png = os.path.join(out_dir, f"eval_{base}.png")
        plt.savefig(out_png, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"[EVAL] Saved: {out_png}")

    return rewards, lengths


if __name__ == "__main__":
    main(
        ckpt_path=CKPT_PATH,
        n_episodes=100,
        max_steps=200,
        sim_time=20.0,
        dt=0.5,
        num_points=15,
        rho=5000.0,
        seed=0,
        render=True,
    )
    print("ok")
