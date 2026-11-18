from gymnasium.envs.registration import register

from pancake_flip.env import PancakeFlipEnv

register(
    id="PancakeFlip-v0",
    entry_point="pancake_flip.env:PancakeFlipEnv",
    max_episode_steps=250,
)

__all__ = ["PancakeFlipEnv"]
