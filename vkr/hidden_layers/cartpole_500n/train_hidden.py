"""
Обучение скрытого слоя актора для CartPole (DiehlAndCook2015v2)
с энкодером InputLayer (60 входных нейронов, фиксированные диапазоны).

Что делает:
  - Обучает n_neurons нейронов через STDP на оффлайн-датасете
  - После каждой эпохи оценивает сеть в CartPole-v1 (prototype policy)
  - Сохраняет лучший чекпойнт (по success_rate, затем mean_length)
  - Сохраняет финальный чекпойнт после каждой эпохи
  - Пишет JSON-лог и строит кривые обучения

Все гиперпараметры настраиваются в блоке if __name__ == "__main__".
"""

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, REPO_ROOT)  # local modified bindsnet

import json
import csv
import random
import warnings
from time import time as now
from typing import Dict, List, Optional

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gymnasium as gym

from sklearn.model_selection import train_test_split

from bindsnet.encoding import poisson
from bindsnet.network import load as bindsnet_load
from bindsnet.network.monitors import Monitor
from bindsnet.models import DiehlAndCook2015v2
from bindsnet.evaluation import assign_labels, all_activity, proportion_weighting

warnings.filterwarnings("ignore", category=UserWarning)


# =========================================================
# INPUT LAYER
# =========================================================
class InputLayer:
    """
    GRF-энкодер: 4 фичи × 15 центров = 60 входных нейронов.
    Диапазоны фиксированы, без per-feature нормализации.
    """

    def __init__(self, rho: float = 5000.0):
        self.num_features = 4
        self.rho = rho
        self.h = np.zeros(self.num_features)
        self.num_points = 15

        self.x_values         = np.linspace(-2.5,  2.5,  self.num_points)
        self.h[0]             = self.x_values[1] - self.x_values[0]

        self.vel_values       = np.linspace(-1.6,  1.6,  self.num_points)
        self.h[1]             = self.vel_values[1] - self.vel_values[0]

        self.theta_values     = np.linspace(-0.45, 0.45, self.num_points)
        self.h[2]             = self.theta_values[1] - self.theta_values[0]

        self.theta_vel_values = np.linspace(-2.6,  2.6,  self.num_points)
        self.h[3]             = self.theta_vel_values[1] - self.theta_vel_values[0]

        self.h = self.h.reshape(self.num_features, 1)

        self.values = np.vstack((
            self.x_values,
            self.vel_values,
            self.theta_values,
            self.theta_vel_values,
        ))

        self.n_inpt = self.num_features * self.num_points  # 60

    def step(self, state, device: str = "cpu") -> torch.Tensor:
        state = np.array(state, dtype=np.float32).reshape(self.num_features, 1)
        diff  = (self.values - state) ** 2
        exp   = np.exp(-0.5 * diff / self.h ** 2)
        return (
            self.rho
            * torch.from_numpy(exp.flatten()).type(torch.FloatTensor).to(device)
        )


# =========================================================
# УТИЛИТЫ
# =========================================================
def ensure_dirs(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# =========================================================
# CARTPOLE EVALUATION — prototype policy
# =========================================================
def _cosine_sim(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> float:
    num = float(torch.dot(a.float(), b.float()).item())
    den = float((a.float().norm() * b.float().norm()).item()) + eps
    return num / den


def build_action_prototypes(
    network,
    encoder: InputLayer,
    states_np: np.ndarray,
    actions_np: np.ndarray,
    time_sec: float,
    dt: float,
    device: str,
    n_per_action: int = 200,
    seed: int = 0,
) -> dict:
    rng = np.random.RandomState(seed)
    timesteps = int(round(time_sec / dt))
    protos = {}

    for a in [0, 1]:
        idx  = np.where(actions_np == a)[0]
        take = min(n_per_action, len(idx))
        pick = rng.choice(idx, size=take, replace=False)

        acc = None
        for j in pick:
            rates = encoder.step(states_np[j], device=device)
            spk_X = poisson(datum=rates, time=timesteps, dt=dt)

            network.reset_state_variables()
            network.run(inputs={"X": spk_X}, time=time_sec, learning=False, one_step=False)

            Y_spk  = network.monitors["Y_spikes"].get("s")
            counts = Y_spk.sum(dim=0)[0].detach().float()
            acc    = counts if acc is None else (acc + counts)

        proto = (acc / float(take)).detach().float()
        proto = proto / (proto.norm() + 1e-8)
        protos[a] = proto

    return protos


@torch.no_grad()
def evaluate_in_cartpole(
    network,
    encoder: InputLayer,
    time_sec: float,
    dt: float,
    states_np: np.ndarray,
    actions_np: np.ndarray,
    n_episodes: int = 50,
    max_steps: int = 500,
    proto_per_action: int = 200,
    seed: int = 0,
    device: str = "cpu",
    epoch: int = 0,
) -> dict:
    print(
        f"\n  [CartPole eval, epoch {epoch}] "
        f"Строю прототипы ({proto_per_action}/action)...",
        flush=True,
    )

    protos    = build_action_prototypes(
        network=network,
        encoder=encoder,
        states_np=states_np,
        actions_np=actions_np,
        time_sec=time_sec,
        dt=dt,
        device=device,
        n_per_action=proto_per_action,
        seed=seed,
    )
    timesteps = int(round(time_sec / dt))
    env       = gym.make("CartPole-v1", render_mode=None)

    ep_lengths   = []
    ep_successes = []

    for ep in range(n_episodes):
        state, _ = env.reset(seed=seed + ep)
        network.reset_state_variables()
        success = False

        for step in range(1, max_steps + 1):
            rates = encoder.step(state, device=device)
            spk_X = poisson(datum=rates, time=timesteps, dt=dt)

            network.run(inputs={"X": spk_X}, time=time_sec, learning=False, one_step=False)
            Y_spk  = network.monitors["Y_spikes"].get("s")
            counts = Y_spk.sum(dim=0)[0]

            v = counts.float()
            if v.norm() > 0:
                v = v / (v.norm() + 1e-8)
            sims   = {a: _cosine_sim(v, protos[a]) for a in protos}
            action = int(max(sims, key=lambda k: sims[k]))

            next_state, _, terminated, truncated, _ = env.step(action)
            state = next_state
            network.reset_state_variables()

            if truncated:
                success = True
                break
            if terminated:
                break

        ep_lengths.append(step)
        ep_successes.append(int(success))

    env.close()

    mean_len     = float(np.mean(ep_lengths))
    success_rate = float(np.mean(ep_successes))
    max_len      = int(np.max(ep_lengths))

    print(
        f"  mean_len={mean_len:.1f}  max_len={max_len}  "
        f"success={success_rate:.2f}  ({n_episodes} episodes)",
        flush=True,
    )

    network.reset_state_variables()

    return {
        "epoch":        epoch,
        "n_episodes":   n_episodes,
        "ep_lengths":   ep_lengths,
        "ep_successes": ep_successes,
        "mean_length":  mean_len,
        "max_length":   max_len,
        "success_rate": success_rate,
    }


# =========================================================
# КРИВЫЕ ОБУЧЕНИЯ — вспомогалки
# =========================================================
def _update_acc(
    spike_record: torch.Tensor,
    current_actions: torch.Tensor,
    assignments: torch.Tensor,
    proportions: torch.Tensor,
    n_classes: int,
    acc_curves: Dict[str, list],
):
    for scheme in acc_curves:
        if scheme == "all":
            pred = all_activity(spike_record, assignments, n_classes)
        else:
            pred = proportion_weighting(spike_record, assignments, proportions, n_classes)
        acc = torch.sum(current_actions.long() == pred).float() / len(current_actions)
        acc_curves[scheme].append(100.0 * acc.item())


# =========================================================
# ОСНОВНАЯ ФУНКЦИЯ ОБУЧЕНИЯ
# =========================================================
def train(
    # --- архитектура ---
    n_neurons: int = 100,
    inhib: float = 0.1,
    w_init_hid: float = 0.5,
    thresh: float = -58.0,
    refrac: int = 5,
    wmin: float = 0.0,
    wmax: float = 1.0,
    # --- обучение ---
    seed: int = 0,
    n_epochs: int = 10,
    n_train: int = 15600,
    n_test: int = 3900,
    n_states_cap: int = 20000,
    lr: float = 1e-5,
    lr_decay: float = 1.0,
    theta_plus: float = 0.0,
    tc_theta_decay: float = 1e7,
    time: int = 20,
    dt: float = 0.5,
    update_interval: int = 50,
    progress_interval: int = 500,
    # --- энкодер ---
    rho: float = 5000.0,
    # --- оценка в среде ---
    eval_n_episodes: int = 50,
    eval_proto_per_action: int = 200,
    eval_max_steps: int = 500,
    # --- пути ---
    root_dir: str = SCRIPT_DIR,
    run_name: Optional[str] = None,
    gpu: bool = False,
):
    assert n_train % update_interval == 0, "n_train must be divisible by update_interval"

    device = "cuda" if gpu and torch.cuda.is_available() else "cpu"
    set_seed(seed)

    # ---- Автоматическое имя run ----
    if run_name is None:
        run_name = (
            f"n{n_neurons}_lr{lr}_inh{inhib}_"
            f"thr{thresh}_ref{refrac}_rho{int(rho)}_ep{n_epochs}"
        )

    run_dir = os.path.join(root_dir, "runs", run_name)
    ensure_dirs(run_dir)

    print("=" * 60)
    print(f"RUN: {run_name}")
    print(f"  n_neurons={n_neurons}  lr={lr}  inhib={inhib}")
    print(f"  thresh={thresh}  refrac={refrac}  rho={rho}")
    print(f"  n_epochs={n_epochs}  n_train={n_train}")
    print(f"  device={device}  seed={seed}")
    print(f"  run_dir={run_dir}")
    print("=" * 60)

    # ---- Датасет ----
    data_path = os.path.join(REPO_ROOT, "data", "states_for_scale.json")
    with open(data_path, "r") as f:
        raw = json.load(f)

    n_states   = min(len(raw["state"]),  n_states_cap)
    state_arr  = np.array(raw["state"][:n_states],  dtype=np.float32)
    action_arr = np.array(raw["action"][:n_states], dtype=np.int64)

    X = torch.tensor(state_arr,  dtype=torch.float32)
    y = torch.tensor(action_arr, dtype=torch.long)

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=n_test, shuffle=True, random_state=seed, stratify=y
    )
    print(f"Dataset: total={n_states}, train={len(X_train)}, "
          f"using first {n_train} per epoch\n")

    # ---- Энкодер ----
    encoder = InputLayer(rho=rho)

    # ---- Сеть ----
    n_classes  = 2
    timesteps  = int(time / dt)

    network = DiehlAndCook2015v2(
        n_inpt=encoder.n_inpt,
        n_neurons=n_neurons,
        inh=inhib,
        dt=dt,
        norm=None,
        theta_plus=theta_plus,
        tc_theta_decay=tc_theta_decay,
        nu=[0.25 * lr, lr],
        w_init_hid=w_init_hid,
        thresh=thresh,
        refrac=refrac,
        wmin=wmin,
        wmax=wmax,
    )

    mon_Y = Monitor(network.layers["Y"], state_vars=["s"], time=timesteps)
    network.add_monitor(mon_Y, name="Y_spikes")

    # ---- Состояние меток ----
    assignments  = -torch.ones(n_neurons, dtype=torch.long)
    proportions  = torch.zeros(n_neurons, n_classes)
    rates        = torch.zeros(n_neurons, n_classes)

    # ---- История ----
    history_acc = {"all": [], "proportion": []}   # per-epoch mean accuracy
    history_env: List[dict] = []

    best_score  = (-1.0, -1.0)   # (success_rate, mean_length) — maximize both
    best_epoch  = 0

    global_start = now()

    for epoch in range(1, n_epochs + 1):
        print(f"\n{'='*60}")
        print(f"  Epoch {epoch}/{n_epochs}")
        print(f"{'='*60}")

        spike_record = torch.zeros(update_interval, timesteps, n_neurons)
        acc_curves   = {"all": [], "proportion": []}
        t_start      = now()

        for i in range(n_train):
            # Прогресс
            if i > 0 and i % progress_interval == 0:
                print(f"    {i}/{n_train}  ({now() - t_start:.1f}s)", flush=True)
                t_start = now()

            # Обновление меток и точности каждые update_interval шагов
            if i > 0 and i % update_interval == 0:
                network.connections["X", "Y"].update_rule.nu[1] *= lr_decay

                current_actions = y_train[i - update_interval: i]
                _update_acc(spike_record, current_actions,
                            assignments, proportions, n_classes, acc_curves)
                assignments, proportions, rates = assign_labels(
                    spike_record, current_actions, n_classes, rates
                )

            # Прямой проход
            state     = X_train[i % len(X_train)]
            rates_enc = encoder.step(state.cpu().numpy(), device=device)
            sample    = poisson(datum=rates_enc, time=time, dt=dt)
            network.run(inputs={"X": sample}, time=time, one_step=False)

            y_spikes = network.monitors["Y_spikes"].get("s").squeeze(1)
            spike_record[i % update_interval] = y_spikes
            network.reset_state_variables()

        print(f"    {n_train}/{n_train}  ({now() - t_start:.1f}s)", flush=True)

        # Финальный update в конце эпохи
        current_actions = y_train[n_train - update_interval: n_train]
        _update_acc(spike_record, current_actions,
                    assignments, proportions, n_classes, acc_curves)
        assignments, proportions, rates = assign_labels(
            spike_record, current_actions, n_classes, rates
        )

        # Средние точности за эпоху
        mean_all  = float(np.mean(acc_curves["all"]))        if acc_curves["all"]        else 0.0
        mean_prop = float(np.mean(acc_curves["proportion"])) if acc_curves["proportion"] else 0.0
        print(f"\n  Epoch {epoch} accuracy:  all={mean_all:.1f}%  "
              f"proportion={mean_prop:.1f}%")

        history_acc["all"].append(mean_all)
        history_acc["proportion"].append(mean_prop)

        # Сохраняем финальный чекпойнт эпохи
        final_pt = os.path.join(run_dir, f"epoch_{epoch:03d}_final.pt")
        network.save(final_pt)
        aux_pt = os.path.join(run_dir, f"epoch_{epoch:03d}_aux.pt")
        torch.save((assignments, proportions, rates), aux_pt)

        # Оценка в среде
        eval_result = evaluate_in_cartpole(
            network=network,
            encoder=encoder,
            time_sec=float(time),
            dt=dt,
            states_np=state_arr,
            actions_np=action_arr,
            n_episodes=eval_n_episodes,
            max_steps=eval_max_steps,
            proto_per_action=eval_proto_per_action,
            seed=seed,
            device=device,
            epoch=epoch,
        )
        eval_result["mean_acc_all"]        = mean_all
        eval_result["mean_acc_proportion"] = mean_prop
        history_env.append(eval_result)

        # Лучший чекпойнт
        score = (eval_result["success_rate"], eval_result["mean_length"])
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_pt  = os.path.join(run_dir, "best.pt")
            best_aux = os.path.join(run_dir, "best_aux.pt")
            network.save(best_pt)
            torch.save((assignments, proportions, rates), best_aux)
            print(f"  >> Новый лучший чекпойнт (epoch {epoch}, "
                  f"success={score[0]:.2f}, mean_len={score[1]:.1f})")

        # Сохраняем лог после каждой эпохи (не теряем при прерывании)
        _save_log(run_dir, run_name, history_acc, history_env, best_epoch,
                  n_neurons, lr, inhib, rho, n_epochs, n_train)
        _save_plot(run_dir, history_acc, history_env, n_neurons, n_epochs)

    # ---- Итоговая сводка ----
    total_time = now() - global_start
    print(f"\n{'='*60}")
    print("ОБУЧЕНИЕ ЗАВЕРШЕНО")
    print(f"{'='*60}")
    print(f"Общее время: {total_time:.1f}s ({total_time / 60:.1f} min)")
    print(f"\n{'Epoch':>6}  {'mean_len':>9}  {'max_len':>8}  {'success':>8}  {'acc_all%':>9}")
    print("-" * 48)
    for r in history_env:
        marker = " <-- best" if r["epoch"] == best_epoch else ""
        print(
            f"{r['epoch']:>6}  {r['mean_length']:>9.1f}  "
            f"{r['max_length']:>8}  {r['success_rate']:>8.2f}  "
            f"{r['mean_acc_all']:>9.1f}{marker}"
        )
    print(f"\nЛучший чекпойнт: epoch {best_epoch}  "
          f"success={best_score[0]:.2f}  mean_len={best_score[1]:.1f}")
    print(f"Файлы сохранены в: {run_dir}")


# =========================================================
# СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
# =========================================================
def _save_log(
    run_dir: str,
    run_name: str,
    history_acc: dict,
    history_env: list,
    best_epoch: int,
    n_neurons: int,
    lr: float,
    inhib: float,
    rho: float,
    n_epochs: int,
    n_train: int,
):
    log = {
        "run_name":   run_name,
        "n_neurons":  n_neurons,
        "lr":         lr,
        "inhib":      inhib,
        "rho":        rho,
        "n_epochs":   n_epochs,
        "n_train":    n_train,
        "best_epoch": best_epoch,
        "history_acc": history_acc,
        "history_env": history_env,
    }
    with open(os.path.join(run_dir, "run_log.json"), "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)

    # CSV для быстрого просмотра
    csv_path = os.path.join(run_dir, "epoch_summary.csv")
    fieldnames = [
        "epoch", "mean_length", "max_length", "success_rate",
        "mean_acc_all", "mean_acc_proportion",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in history_env:
            writer.writerow({k: r.get(k, "") for k in fieldnames})


def _save_plot(
    run_dir: str,
    history_acc: dict,
    history_env: list,
    n_neurons: int,
    n_epochs: int,
):
    epochs = [r["epoch"] for r in history_env]
    if not epochs:
        return

    mean_lens     = [r["mean_length"]  for r in history_env]
    success_rates = [r["success_rate"] for r in history_env]
    acc_all       = history_acc["all"]
    acc_prop      = history_acc["proportion"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle(f"Training: n_neurons={n_neurons}, epochs={n_epochs}", fontsize=13)

    ax = axes[0, 0]
    ax.plot(epochs, acc_all,  marker="o", label="all_activity")
    ax.plot(epochs, acc_prop, marker="s", label="proportion")
    ax.set_xlabel("epoch");  ax.set_ylabel("accuracy, %")
    ax.set_title("Train accuracy (STDP)")
    ax.legend();  ax.grid(True, alpha=0.4)

    ax = axes[0, 1]
    ax.plot(epochs, mean_lens, marker="o", color="steelblue")
    ax.axhline(500, color="green", linestyle="--", linewidth=1, label="max_steps=500")
    ax.set_xlabel("epoch");  ax.set_ylabel("mean episode length")
    ax.set_title("CartPole — mean episode length")
    ax.legend();  ax.grid(True, alpha=0.4)

    ax = axes[1, 0]
    ax.plot(epochs, success_rates, marker="o", color="seagreen")
    ax.set_xlabel("epoch");  ax.set_ylabel("success rate")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"CartPole — success rate ({history_env[0]['n_episodes']} eps)")
    ax.grid(True, alpha=0.4)

    ax = axes[1, 1]
    max_lens = [r["max_length"] for r in history_env]
    ax.plot(epochs, max_lens, marker="o", color="tomato")
    ax.axhline(500, color="green", linestyle="--", linewidth=1, label="max_steps=500")
    ax.set_xlabel("epoch");  ax.set_ylabel("max episode length")
    ax.set_title("CartPole — max episode length")
    ax.legend();  ax.grid(True, alpha=0.4)

    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "training_curves.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


# =========================================================
# ЗАПУСК
# =========================================================
if __name__ == "__main__":
    train(
        # --- архитектура ---
        n_neurons   = 500,       # варьировать: 1, 2, 5, 10, 25, 50, ...
        inhib       = 0.1,
        w_init_hid  = 0.5,
        thresh      = -58.0,
        refrac      = 5,
        wmin        = 0.0,
        wmax        = 1.0,

        # --- обучение ---
        seed            = 0,
        n_epochs        = 100,    # сколько эпох обучать
        n_train         = 15600,
        n_test          = 3900,
        n_states_cap    = 20000,
        lr              = 1e-5,
        lr_decay        = 1.0,
        theta_plus      = 0.0,
        tc_theta_decay  = 1e7,
        time            = 20,
        dt              = 0.5,
        update_interval = 10,
        progress_interval = 500,

        # --- энкодер ---
        rho = 5000.0,

        # --- оценка в среде ---
        eval_n_episodes     = 50,
        eval_proto_per_action = 200,
        eval_max_steps      = 500,

        # --- пути ---
        root_dir = SCRIPT_DIR,
        run_name = None,   # None = автоматически из гиперпараметров
        gpu      = False,
    )
