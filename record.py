"""Watch or record a trained agent.

python record.py runs/sac_dense_s0/best_model.zip --algo sac --human
python record.py runs/sac_dense_s0/best_model.zip --algo sac --out flip.gif
"""

import argparse

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np

import pancake_flip  # noqa: F401


def record_episode(env, policy, path, seed=0, only_if_success=False, max_steps=250):
    """Run one episode with policy(obs) -> action and save frames to path."""
    obs, _ = env.reset(seed=seed)
    if hasattr(policy, "reset"):
        policy.reset()
    frames = [env.render()]
    info = {}
    for _ in range(max_steps):
        obs, r, term, trunc, info = env.step(policy(obs))
        frames.append(env.render())
        if term or trunc:
            break
    if only_if_success and not info.get("is_success"):
        return False
    frames += [frames[-1]] * 15  # hold the last frame a moment
    fps = env.metadata.get("render_fps", 50)
    if path.endswith(".gif"):
        imageio.mimsave(path, frames[::2], duration=2000 / fps, loop=0)
    else:
        imageio.mimsave(path, frames, fps=fps)
    return bool(info.get("is_success"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--algo", choices=["ppo", "sac"], default="sac")
    ap.add_argument("--out", default="agent.gif")
    ap.add_argument("--human", action="store_true")
    ap.add_argument("--episodes", type=int, default=5)
    args = ap.parse_args()

    from stable_baselines3 import PPO, SAC
    model = (PPO if args.algo == "ppo" else SAC).load(args.model)
    policy = lambda obs: model.predict(obs, deterministic=True)[0]

    if args.human:
        env = gym.make("PancakeFlip-v0", render_mode="human")
        for ep in range(args.episodes):
            obs, _ = env.reset(seed=ep)
            done = False
            while not done:
                obs, r, term, trunc, info = env.step(policy(obs))
                done = term or trunc
            print(f"episode {ep}: success={info['is_success']} dropped={info['dropped']}")
    else:
        env = gym.make("PancakeFlip-v0", render_mode="rgb_array", width=640, height=480)
        ok = record_episode(env, policy, args.out)
        print(f"saved {args.out} (success={ok})")


if __name__ == "__main__":
    main()
