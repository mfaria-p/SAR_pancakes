"""Train an agent on PancakeFlip-v0 with Stable-Baselines3.

Examples
--------
python train.py --algo sac --steps 1000000
python train.py --algo ppo --steps 3000000 --n-envs 8
python train.py --algo sac --reward-mode sparse          # shaping ablation
python train.py --algo sac --watch                       # live window with the latest policy
python train.py --algo sac --videos --show-every 100000  # save mp4s of progress

Logs go to runs/<name>/ (TensorBoard: tensorboard --logdir runs).
"""

import os

# Silence TensorFlow's startup messages (TensorBoard may import it in every
# worker process). Must be set before anything imports TensorFlow.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import argparse
import subprocess
import sys
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import (
    BaseCallback, CheckpointCallback, EvalCallback,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

import pancake_flip  # noqa: F401  (registers PancakeFlip-v0)


class SaveLatestCallback(BaseCallback):
    """Save the current policy to runs/<name>/latest_model.zip every `every` steps.

    watch.py reloads that file, so the live window always shows the newest policy
    without pausing training.
    """

    def __init__(self, path, every):
        super().__init__()
        self.path = Path(path)
        self.every = every
        self._last = -every

    def _save(self):
        tmp = self.path.with_name("latest_model_tmp.zip")
        self.model.save(tmp)
        for _ in range(20):  # the viewer may be reading the old file for a moment
            try:
                os.replace(tmp, self.path)
                break
            except PermissionError:
                import time
                time.sleep(0.1)

    def _on_training_start(self):
        self._save()  # untrained policy, so the window shows something right away

    def _on_step(self):
        if self.num_timesteps - self._last >= self.every:
            self._last = self.num_timesteps
            self._save()
        return True


class VideoCallback(BaseCallback):
    """Every `every` steps, record one episode to runs/<name>/videos/step_<N>.mp4."""

    def __init__(self, every, video_dir):
        super().__init__()
        self.every = every
        self.video_dir = Path(video_dir)
        self._last = 0
        self._env = None

    def _record(self):
        from record import record_episode
        if self._env is None:
            self.video_dir.mkdir(parents=True, exist_ok=True)
            self._env = gym.make("PancakeFlip-v0", render_mode="rgb_array", width=640, height=480)
        step = self.num_timesteps
        path = str(self.video_dir / f"step_{step:08d}.mp4")
        policy = lambda obs: self.model.predict(obs, deterministic=True)[0]
        record_episode(self._env, policy, path, seed=step)
        print(f"[video] saved {path}")

    def _on_training_start(self):
        self._record()

    def _on_step(self):
        if self.num_timesteps - self._last >= self.every:
            self._last = self.num_timesteps
            self._record()
        return True

    def _on_training_end(self):
        if self.num_timesteps != self._last:
            self._record()
        if self._env is not None:
            self._env.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", choices=["ppo", "sac"], default="sac")
    p.add_argument("--steps", type=int, default=1_000_000)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--reward-mode", choices=["dense", "sparse"], default="dense")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--name", default=None)
    p.add_argument("--watch", action="store_true",
                   help="open a live MuJoCo window that keeps playing the latest policy")
    p.add_argument("--videos", action="store_true",
                   help="save an mp4 of one episode every --show-every steps")
    p.add_argument("--show-every", type=int, default=50_000,
                   help="steps between videos")
    p.add_argument("--save-latest-every", type=int, default=10_000,
                   help="steps between policy updates shown in the live window")
    args = p.parse_args()

    name = args.name or f"{args.algo}_{args.reward_mode}_s{args.seed}"
    out = Path("runs") / name
    out.mkdir(parents=True, exist_ok=True)

    env_kwargs = {"reward_mode": args.reward_mode}
    n_envs = args.n_envs if args.algo == "ppo" else min(args.n_envs, 4)
    env = make_vec_env(
        "PancakeFlip-v0", n_envs=n_envs, seed=args.seed,
        env_kwargs=env_kwargs, vec_env_cls=SubprocVecEnv,
    )
    # Evaluate on the dense env so "is_success" and episode return are comparable.
    eval_env = make_vec_env("PancakeFlip-v0", n_envs=1, seed=args.seed + 1000)

    if args.algo == "ppo":
        model = PPO(
            "MlpPolicy", env, n_steps=1024, batch_size=256, n_epochs=10,
            gamma=0.99, gae_lambda=0.95, learning_rate=3e-4, ent_coef=0.0,
            clip_range=0.2, policy_kwargs=dict(net_arch=[256, 256]),
            tensorboard_log=str(out), seed=args.seed, verbose=1,
        )
    else:
        model = SAC(
            "MlpPolicy", env, buffer_size=1_000_000, batch_size=256,
            learning_rate=3e-4, gamma=0.99, tau=0.005, train_freq=1,
            gradient_steps=1, learning_starts=10_000,
            policy_kwargs=dict(net_arch=[256, 256]),
            tensorboard_log=str(out), seed=args.seed, verbose=1,
        )

    callbacks = [
        EvalCallback(
            eval_env, n_eval_episodes=20, eval_freq=max(25_000 // n_envs, 1),
            best_model_save_path=str(out), log_path=str(out), deterministic=True,
        ),
        CheckpointCallback(save_freq=max(200_000 // n_envs, 1), save_path=str(out / "ckpt")),
        SaveLatestCallback(out / "latest_model.zip", args.save_latest_every),
    ]
    if args.videos:
        callbacks.append(VideoCallback(args.show_every, out / "videos"))

    viewer = None
    if args.watch:
        here = Path(__file__).resolve().parent
        viewer = subprocess.Popen(
            [sys.executable, str(here / "watch.py"), str(out), "--algo", args.algo]
        )
        print(f"[watch] live viewer started (pid {viewer.pid}). "
              f"You can also run it yourself: python watch.py {out} --algo {args.algo}")

    try:
        model.learn(total_timesteps=args.steps, callback=callbacks, progress_bar=False)
        model.save(out / "final_model")
        print(f"Saved to {out}")
    finally:
        if viewer is not None and viewer.poll() is None:
            viewer.terminate()


if __name__ == "__main__":
    main()
