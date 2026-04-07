"""
PPO training script for BIM Clash/Conflict Resolution.

Trains a PPO agent with stable-baselines3 on BimClashEnv.
The agent learns to move MEP elements to resolve clashes with structural
elements purely from the reward signal — no hand-crafted heuristics.

Usage
-----
    # Install deps first (once):
    pip install stable-baselines3[extra] gymnasium

    # Train on medium task (default):
    python -m bim_env.train

    # Override settings via environment variables:
    TASK=hard TOTAL_TIMESTEPS=500000 N_ENVS=8 python -m bim_env.train

Outputs
-------
    models/ppo_bim_<task>/              Checkpoints every SAVE_FREQ steps
    models/ppo_bim_<task>/best/         Best model by eval reward
    models/ppo_bim_<task>/ppo_bim_final.zip   Final saved model
    logs/ppo_bim_<task>/                TensorBoard logs

View training curves:
    tensorboard --logdir logs/
"""

import os
import time
from pathlib import Path

import numpy as np

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import (
        BaseCallback,
        CheckpointCallback,
        EvalCallback,
    )
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import VecNormalize
except ImportError as exc:
    raise ImportError(
        "stable-baselines3 is required. Install with:\n"
        "  pip install stable-baselines3[extra] gymnasium"
    ) from exc

try:
    from .gym_env import BimClashEnv
except ImportError:
    from gym_env import BimClashEnv

# ---------------------------------------------------------------------------
# Configuration (all overridable via environment variables)
# ---------------------------------------------------------------------------
TASK            = os.getenv("TASK", "medium")
TOTAL_TIMESTEPS = int(os.getenv("TOTAL_TIMESTEPS", "200000"))
N_ENVS          = int(os.getenv("N_ENVS", "4"))
EVAL_FREQ       = int(os.getenv("EVAL_FREQ", "10000"))
SAVE_FREQ       = int(os.getenv("SAVE_FREQ", "25000"))
SEED            = int(os.getenv("SEED", "42"))

MODEL_DIR = Path(f"models/ppo_bim_{TASK}")
LOG_DIR   = Path(f"logs/ppo_bim_{TASK}")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

class ProgressCallback(BaseCallback):
    """Prints a one-line training summary every PRINT_FREQ steps."""

    def __init__(self, print_freq: int = 5_000):
        super().__init__()
        self.print_freq = print_freq
        self._last_print = 0
        self._ep_grades: list = []
        self._ep_clashes: list = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            # SB3 wraps terminal info in 'final_info' for VecEnv
            final = info.get("final_info") or info
            if "grade" in final:
                self._ep_grades.append(final["grade"])
                self._ep_clashes.append(final["num_clashes"])

        if self.num_timesteps - self._last_print >= self.print_freq:
            self._last_print = self.num_timesteps
            if self._ep_grades:
                success = sum(g >= 0.70 for g in self._ep_grades)
                print(
                    f"  [{self.num_timesteps:>8,} steps]  "
                    f"episodes={len(self._ep_grades):4d}  "
                    f"mean_grade={np.mean(self._ep_grades):.3f}  "
                    f"success_rate={success/len(self._ep_grades):.0%}  "
                    f"mean_clashes={np.mean(self._ep_clashes):.1f}",
                    flush=True,
                )
                self._ep_grades.clear()
                self._ep_clashes.clear()
        return True


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train() -> None:
    print(
        f"\n{'='*60}\n"
        f"  BIM Clash RL Training\n"
        f"  Task:       {TASK}\n"
        f"  Timesteps:  {TOTAL_TIMESTEPS:,}\n"
        f"  Parallel envs: {N_ENVS}\n"
        f"{'='*60}\n"
    )

    # ── Vectorised training environments ────────────────────────────────────
    train_env = make_vec_env(
        lambda: Monitor(BimClashEnv(task=TASK)),
        n_envs=N_ENVS,
        seed=SEED,
    )
    train_env = VecNormalize(
        train_env, norm_obs=True, norm_reward=True, clip_obs=5.0
    )

    # ── Evaluation environment (reward not normalised for readable grades) ──
    eval_env = make_vec_env(
        lambda: Monitor(BimClashEnv(task=TASK)),
        n_envs=1,
        seed=SEED + 999,
    )
    eval_env = VecNormalize(
        eval_env, training=False, norm_obs=True, norm_reward=False, clip_obs=5.0
    )

    # ── PPO agent ───────────────────────────────────────────────────────────
    # Observation: 112-dim  |  Action: 15-dim continuous (5 elements × 3 axes)
    model = PPO(
        "MlpPolicy",
        train_env,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs={"net_arch": [256, 256]},
        tensorboard_log=str(LOG_DIR),
        verbose=0,
        seed=SEED,
    )

    callbacks = [
        ProgressCallback(print_freq=5_000),
        CheckpointCallback(
            save_freq=max(SAVE_FREQ // N_ENVS, 1),
            save_path=str(MODEL_DIR),
            name_prefix="ppo_bim",
            save_vecnormalize=True,
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=str(MODEL_DIR / "best"),
            log_path=str(LOG_DIR),
            eval_freq=max(EVAL_FREQ // N_ENVS, 1),
            n_eval_episodes=10,
            deterministic=True,
            verbose=1,
        ),
    ]

    # ── Train ───────────────────────────────────────────────────────────────
    t0 = time.time()
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callbacks, progress_bar=True)
    elapsed = time.time() - t0

    # ── Save final model + normalisation stats ──────────────────────────────
    final_path = MODEL_DIR / "ppo_bim_final"
    model.save(str(final_path))
    train_env.save(str(MODEL_DIR / "vec_normalize.pkl"))

    print(f"\nTraining complete in {elapsed:.0f}s")
    print(f"Model:      {final_path}.zip")
    print(f"TensorBoard: tensorboard --logdir {LOG_DIR}")

    # ── Final evaluation (raw env, no normalisation) ─────────────────────────
    print(f"\n{'─'*50}")
    print(f"Final evaluation — 10 episodes on task='{TASK}'")
    print(f"{'─'*50}")

    eval_env_raw = BimClashEnv(task=TASK)
    grades = []
    for ep in range(10):
        obs, _ = eval_env_raw.reset()
        done = False
        while not done:
            # Wrap obs for VecNormalize — use raw model predict for simplicity
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env_raw.step(action)
            done = terminated or truncated
        grades.append(info["grade"])
        status = "✓ resolved" if info["grade"] >= 0.70 else "✗ partial"
        print(
            f"  Ep {ep+1:2d}: grade={info['grade']:.3f}  "
            f"clashes_left={info['num_clashes']}  steps={info['step']}  {status}"
        )

    success_rate = sum(g >= 0.70 for g in grades) / len(grades)
    print(f"\nMean grade:   {np.mean(grades):.3f}")
    print(f"Success rate: {success_rate:.0%}  (grade ≥ 0.70 = all clashes resolved)")


if __name__ == "__main__":
    train()
