"""Live viewer: keeps playing episodes with the newest saved policy of a training run.

Run it next to training (train.py --watch starts it for you):

    python watch.py runs/sac_dense_s0 --algo sac

It reloads runs/<name>/latest_model.zip whenever training saves a newer one
(every 10k steps by default), so you can watch the agent improve live.
Close the window or press Ctrl+C to stop; training is not affected.
"""

import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import gymnasium as gym

import pancake_flip  # noqa: F401


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="e.g. runs/sac_dense_s0")
    ap.add_argument("--algo", choices=["ppo", "sac"], default="sac")
    ap.add_argument("--file", default="latest_model.zip")
    ap.add_argument("--headless", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()

    from stable_baselines3 import PPO, SAC
    Algo = PPO if args.algo == "ppo" else SAC
    path = Path(args.run_dir) / args.file

    print(f"[watch] waiting for {path} ...")
    while not path.exists():
        time.sleep(1)

    env = gym.make("PancakeFlip-v0", render_mode=None if args.headless else "human")
    model, loaded_mtime, episode = None, None, 0
    try:
        while True:
            mtime = path.stat().st_mtime
            if mtime != loaded_mtime:
                try:
                    model = Algo.load(path, device="cpu")
                    loaded_mtime = mtime
                    print(f"[watch] loaded policy from training step {model.num_timesteps:,}")
                except Exception:  # file being replaced right now; try again shortly
                    time.sleep(0.5)
                    if model is None:
                        continue

            obs, _ = env.reset(seed=episode)
            done, info = False, {}
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, term, trunc, info = env.step(action)
                done = term or trunc
            episode += 1
            result = "FLIPPED!" if info["is_success"] else ("dropped" if info["dropped"] else "timeout")
            print(f"[watch] episode {episode} (policy @ {model.num_timesteps:,} steps): "
                  f"{result}, best flip {info['best_flip_frac']:.2f}")
            if args.headless and episode >= 3:
                break
    except KeyboardInterrupt:
        pass
    except Exception as e:  # e.g. the window was closed
        print(f"[watch] viewer stopped: {e}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
