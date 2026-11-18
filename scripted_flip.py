"""Hand-tuned open-loop baseline: wind up, flick, then hold a catch pose.

The motion was found with a cross-entropy search over 11 numbers (three arm
poses and two timings). It does not look at the pancake at all, so it only
works when the toss happens to land well. That makes it a useful baseline:
a closed-loop RL policy should beat it.

python scripted_flip.py                 # success rate over 100 episodes
python scripted_flip.py --record flip.gif
python scripted_flip.py --human         # live viewer
"""

import argparse

import gymnasium as gym
import numpy as np

import pancake_flip  # noqa: F401

# wind-up delta (3), flick delta (3), catch delta (3), wind-up time, flick time
PARAMS = np.array([-0.744, -2.059, 0.273, -0.168, -0.432, 0.966,
                   0.297, -0.248, -0.038, 0.232, 0.424])


class ScriptedFlipper:
    def __init__(self, home_action, params=PARAMS):
        self.h = home_action
        self.p = params
        self.t = 0

    def reset(self):
        self.t = 0

    def __call__(self, obs=None):
        p, h, t = self.p, self.h, self.t
        t1 = int(3 + p[9] * 8)
        t2 = int(2 + p[10] * 8)
        if t < t1:
            a = h + p[0:3]
        elif t < t1 + t2:
            a = h + p[3:6]
        else:
            a = h + p[6:9]
        self.t += 1
        return np.clip(a, -1, 1).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--record", default=None, help="save a GIF/MP4 of a successful flip")
    ap.add_argument("--human", action="store_true")
    args = ap.parse_args()

    if args.record:
        from record import record_episode
        # find a seed where the open-loop toss lands (cheap, no rendering) ...
        probe = gym.make("PancakeFlip-v0")
        pol = ScriptedFlipper(probe.unwrapped.home_action)
        good = None
        for seed in range(500):
            probe.reset(seed=seed)
            pol.reset()
            done = False
            while not done:
                _, _, term, trunc, info = probe.step(pol())
                done = term or trunc
            if info["is_success"]:
                good = seed
                break
        if good is None:
            print("no successful seed found")
            return
        # ... then render just that one
        env = gym.make("PancakeFlip-v0", render_mode="rgb_array", width=640, height=480)
        record_episode(env, pol, args.record, seed=good)
        print(f"saved {args.record} (seed {good})")
        return

    env = gym.make("PancakeFlip-v0", render_mode="human" if args.human else None)
    pol = ScriptedFlipper(env.unwrapped.home_action)
    succ, drops = 0, 0
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=ep)
        pol.reset()
        done = False
        while not done:
            obs, r, term, trunc, info = env.step(pol(obs))
            done = term or trunc
        succ += info["is_success"]
        drops += info["dropped"]
    print(f"scripted baseline: success {succ}/{args.episodes}, dropped {drops}/{args.episodes}")


if __name__ == "__main__":
    main()
