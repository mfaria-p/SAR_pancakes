"""PancakeFlip-v0: a MuJoCo robot arm that has to flip a pancake.

A 3-joint planar arm (shoulder, elbow, wrist pitch) holds a frying pan.
The pancake starts in the pan with its golden side DOWN. The agent must
toss it so it lands back in the pan with the golden side UP and settles.

Built on gymnasium.envs.mujoco.MujocoEnv, so it works like the standard
MuJoCo environments (Hopper, HalfCheetah, Reacher...).
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
from gymnasium import spaces, utils
from gymnasium.envs.mujoco import MujocoEnv

XML_PATH = str(Path(__file__).parent / "assets" / "pancake_flip.xml")

DEFAULT_CAMERA_CONFIG = {
    "distance": 2.6,
    "azimuth": 90.0,
    "elevation": -8.0,
    "lookat": np.array([0.45, 0.0, 0.95]),
}

# Arm pose that holds the pan level above the stove.
HOME_QPOS = np.array([-0.6, 1.2, -0.6])


class PancakeFlipEnv(MujocoEnv, utils.EzPickle):
    """Flip a pancake with a robot arm.

    ## Action space
    Box(-1, 1, (3,)): target positions for the shoulder, elbow and wrist,
    scaled to each joint's range. The joints are driven by position servos,
    so the policy says "where to be", and the servo decides the torque.

    ## Observation space (Box, 24 dims)
    | idx   | meaning                                             |
    |-------|-----------------------------------------------------|
    | 0-2   | arm joint angles                                    |
    | 3-5   | arm joint velocities                                |
    | 6-8   | pancake position relative to pan centre (pan frame) |
    | 9-11  | pancake linear velocity (pan frame)                 |
    | 12-14 | pancake angular velocity (world frame)              |
    | 15-17 | pancake "raw side" normal, expressed in pan frame   |
    | 18    | pancake "raw side" normal, world z (+1 = unflipped) |
    | 19-20 | pan tilt as (sin, cos) around the y axis            |
    | 21    | pan centre height above the stove                   |
    | 22    | 1 if the pancake touches the pan, else 0            |
    | 23    | best flip fraction reached so far this episode      |

    ## Rewards (reward_mode="dense")
    * airborne bonus: small reward for pancake height above the pan while it
      is in the air (encourages tossing)
    * flip progress: reward for improving the flip fraction while airborne
    * holding: per-step reward while the pancake is flipped and in the pan
    * centering: tiny reward for keeping the pancake near the pan centre
    * control cost on the action
    * success bonus (+50) when the pancake is flipped, inside the pan and
      at rest for `settle_steps` steps -> episode terminates
    * drop penalty (-10) when the pancake leaves the pan and falls -> terminates

    reward_mode="sparse" gives only the success bonus and drop penalty, which is
    useful as a baseline to show why reward shaping matters.

    ## Episode end
    terminated on success or drop; truncated by TimeLimit (default 250 steps
    = 5 s of simulated time).
    """

    metadata = {"render_modes": ["human", "rgb_array", "depth_array"]}

    def __init__(
        self,
        frame_skip: int = 10,
        reward_mode: str = "dense",
        reset_noise_scale: float = 0.02,
        pancake_mass: float | None = None,
        success_bonus: float = 50.0,
        drop_penalty: float = 10.0,
        ctrl_cost_weight: float = 0.01,
        settle_steps: int = 10,
        default_camera_config: dict | None = None,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self, frame_skip, reward_mode, reset_noise_scale, pancake_mass,
            success_bonus, drop_penalty, ctrl_cost_weight, settle_steps,
            default_camera_config, **kwargs,
        )
        assert reward_mode in ("dense", "sparse")
        self.reward_mode = reward_mode
        self.reset_noise_scale = reset_noise_scale
        self.success_bonus = success_bonus
        self.drop_penalty = drop_penalty
        self.ctrl_cost_weight = ctrl_cost_weight
        self.settle_steps = settle_steps

        obs_space = spaces.Box(-np.inf, np.inf, shape=(24,), dtype=np.float64)
        MujocoEnv.__init__(
            self,
            XML_PATH,
            frame_skip,
            observation_space=obs_space,
            default_camera_config=default_camera_config or DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )
        self.metadata["render_fps"] = int(np.round(1.0 / self.dt))

        # Agent acts in [-1, 1]; we map to the servo control ranges.
        self._ctrl_low = self.model.actuator_ctrlrange[:, 0].copy()
        self._ctrl_high = self.model.actuator_ctrlrange[:, 1].copy()
        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)

        m = self.model
        self._pan_site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "pan_center")
        self._pan_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pan")
        self._pancake_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pancake")
        self._pancake_geoms = {
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)
            for n in ("pancake_raw", "pancake_golden")
        }
        self._pan_geoms = {
            g for g in range(m.ngeom) if m.geom_bodyid[g] == self._pan_body
        }
        self._stove_z = 0.35

        if pancake_mass is not None:
            scale = pancake_mass / m.body_mass[self._pancake_body]
            m.body_mass[self._pancake_body] *= scale
            m.body_inertia[self._pancake_body] *= scale

        self._best_flip = 0.0
        self._settled = 0

    # ------------------------------------------------------------------ helpers
    def _action_to_ctrl(self, action):
        a = np.clip(action, -1.0, 1.0)
        return self._ctrl_low + (a + 1.0) * 0.5 * (self._ctrl_high - self._ctrl_low)

    def _ctrl_to_action(self, ctrl):
        return 2.0 * (ctrl - self._ctrl_low) / (self._ctrl_high - self._ctrl_low) - 1.0

    def _pan_frame(self):
        pos = self.data.site_xpos[self._pan_site].copy()
        rot = self.data.xmat[self._pan_body].reshape(3, 3).copy()
        return pos, rot

    def _pan_velocity(self):
        res = np.zeros(6)
        mujoco.mj_objectVelocity(
            self.model, self.data, mujoco.mjtObj.mjOBJ_SITE, self._pan_site, res, 0
        )
        return res[3:]

    def _pancake_state(self):
        pos = self.data.xpos[self._pancake_body].copy()
        rot = self.data.xmat[self._pancake_body].reshape(3, 3).copy()
        vel = self.data.qvel[3:6].copy()
        angvel = self.data.qvel[6:9].copy()
        return pos, rot, vel, angvel

    def _touching_pan(self):
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if (c.geom1 in self._pancake_geoms and c.geom2 in self._pan_geoms) or (
                c.geom2 in self._pancake_geoms and c.geom1 in self._pan_geoms
            ):
                return True
        return False

    def _touching_anything(self):
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if c.geom1 in self._pancake_geoms or c.geom2 in self._pancake_geoms:
                return True
        return False

    def flip_info(self):
        """Quantities used for reward, termination and logging."""
        pan_pos, pan_rot = self._pan_frame()
        pc_pos, pc_rot, pc_vel, pc_angvel = self._pancake_state()
        rel = pan_rot.T @ (pc_pos - pan_pos)  # pancake in pan frame
        raw_up_world = pc_rot[2, 2]  # world z of the raw side's normal
        flip_frac = (1.0 - raw_up_world) / 2.0  # 0 = raw up, 1 = golden up
        return {
            "rel": rel,
            "dist_xy": float(np.linalg.norm(rel[:2])),
            "height": float(rel[2]),
            "raw_up_world": float(raw_up_world),
            # raw side normal relative to the pan surface: +1 raw up, -1 golden up
            "raw_up_pan": float((pan_rot.T @ pc_rot[:, 2])[2]),
            "flip_frac": float(flip_frac),
            "speed": float(np.linalg.norm(pc_vel - self._pan_velocity())),
            "pc_z": float(pc_pos[2]),
            "pan_pos": pan_pos,
            "pan_rot": pan_rot,
            "pc_rot": pc_rot,
            "pc_vel": pc_vel,
            "pc_angvel": pc_angvel,
        }

    # --------------------------------------------------------------- gym API
    def _get_obs(self):
        f = self.flip_info()
        pan_rot = f["pan_rot"]
        # pan pitch: angle of pan x-axis above horizontal
        pan_x = pan_rot[:, 0]
        pitch = np.arctan2(pan_x[2], pan_x[0])
        return np.concatenate([
            self.data.qpos[:3],
            self.data.qvel[:3],
            f["rel"],
            pan_rot.T @ f["pc_vel"],
            f["pc_angvel"],
            pan_rot.T @ f["pc_rot"][:, 2],
            [f["raw_up_world"]],
            [np.sin(pitch), np.cos(pitch)],
            [f["pan_pos"][2] - self._stove_z],
            [float(self._touching_pan())],
            [self._best_flip],
        ])

    def step(self, action):
        ctrl = self._action_to_ctrl(action)
        self.do_simulation(ctrl, self.frame_skip)

        f = self.flip_info()
        in_pan = f["dist_xy"] < 0.11 and -0.02 < f["height"] < 0.06
        touching = self._touching_pan()
        airborne = not self._touching_anything()
        flipped = f["raw_up_pan"] < -0.9  # golden side facing up, relative to the pan

        # Progress counts only while the pancake is airborne AND above the pan,
        # so flinging it away (it tumbles while falling) earns nothing, and
        # sliding it up the rim to tip it over is not rewarded either.
        progress = 0.0
        over_pan = f["dist_xy"] < 0.13
        if airborne and over_pan and f["flip_frac"] > self._best_flip:
            progress = f["flip_frac"] - self._best_flip
            self._best_flip = f["flip_frac"]
        elif not airborne and f["flip_frac"] > self._best_flip + 0.3:
            self._best_flip = f["flip_frac"]  # count it, but no reward

        if flipped and in_pan and touching and f["speed"] < 0.15:
            self._settled += 1
        else:
            self._settled = 0

        success = self._settled >= self.settle_steps
        dropped = f["pc_z"] < f["pan_pos"][2] - 0.12 or f["pc_z"] < self._stove_z + 0.03

        ctrl_cost = self.ctrl_cost_weight * float(np.sum(np.square(action)))
        r = {
            "airborne": 0.0,
            "progress": 0.0,
            "holding": 0.0,
            "centering": 0.0,
            "ctrl": -ctrl_cost,
            "terminal": 0.0,
        }
        if self.reward_mode == "dense":
            if airborne:
                r["airborne"] = 0.5 * float(np.clip(f["height"], 0.0, 0.3))
            r["progress"] = 10.0 * progress
            if flipped and in_pan:
                r["holding"] = 1.0
            r["centering"] = 0.05 * float(np.exp(-f["dist_xy"] / 0.05))
        if success:
            r["terminal"] = self.success_bonus
        elif dropped:
            r["terminal"] = -self.drop_penalty

        reward = float(sum(r.values()))
        terminated = bool(success or dropped)

        info = {
            "is_success": bool(success),
            "dropped": bool(dropped),
            "flipped": bool(flipped),
            "in_pan": bool(in_pan),
            "flip_frac": f["flip_frac"],
            "best_flip_frac": self._best_flip,
            "pancake_height": f["height"],
            **{f"reward_{k}": v for k, v in r.items()},
        }
        if self.render_mode == "human":
            self.render()
        return self._get_obs(), reward, terminated, False, info

    def reset_model(self):
        noise = self.reset_noise_scale
        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()

        arm = HOME_QPOS + self.np_random.uniform(-noise, noise, size=3)
        qpos[:3] = arm
        self.data.ctrl[:] = arm

        # place the pancake on the pan surface, raw side up, small offset/yaw
        self.set_state(qpos, qvel)
        pan_pos, _ = self._pan_frame()
        xy = self.np_random.uniform(-2 * noise, 2 * noise, size=2)
        yaw = self.np_random.uniform(-np.pi, np.pi)
        qpos[3:6] = pan_pos + np.array([xy[0], xy[1], 0.0085])
        qpos[6:10] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
        self.set_state(qpos, qvel)

        # let things settle for a moment with the arm holding still
        for _ in range(25):
            mujoco.mj_step(self.model, self.data)

        self._best_flip = 0.0
        self._settled = 0
        return self._get_obs()

    @property
    def home_action(self):
        """Action that holds the pan level in the starting pose."""
        return self._ctrl_to_action(HOME_QPOS).astype(np.float32)
