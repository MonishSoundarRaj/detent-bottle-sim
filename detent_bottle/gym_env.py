import gymnasium as gym
import numpy as np
from gymnasium import spaces
from scipy.spatial.transform import Rotation
from .core import DetentBottleSim, SIDES, VIEWS, WORKSPACE, WRISTS

CAMERAS = tuple(VIEWS) + WRISTS
POS_SCALE, ROT_SCALE, MAX_JAW = .02, .10, .09


class DetentBottleEnv(gym.Env):
    """Bimanual detent-latch bottle. Action per arm: [dx, dy, dz, rx, ry, rz, jaw] in [-1, 1]
    (left arm first, 14 values). See README for the full spec."""
    metadata = {'render_modes': ['rgb_array'], 'render_fps': 20}

    def __init__(self, image_size=128, cameras=CAMERAS, max_steps=2400, render_mode=None, bottle_pose=None):
        self.sim = DetentBottleSim(stop_on_success=True, max_steps=max_steps, bottle_pose=bottle_pose)
        self.image_size, self.cameras, self.render_mode = image_size, tuple(cameras), render_mode
        self.action_space = spaces.Box(-1., 1., (14,), np.float32)
        vec = lambda n: spaces.Box(-np.inf, np.inf, (n,), np.float64)
        self.observation_space = spaces.Dict({
            **{c: spaces.Box(0, 255, (image_size, image_size, 3), np.uint8) for c in self.cameras},
            'joint_pos': vec(14), 'joint_vel': vec(14), 'gripper_qpos': vec(4),
            'eef_pos': vec(6), 'eef_quat': vec(8), 'wrist_ft': vec(12), 'finger_force': vec(4)})

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.sim.reset()
        return self._obs(), self._info()

    def step(self, action):
        s = self.sim; a = np.clip(np.asarray(action, float).reshape(2, 7), -1, 1)
        for i in range(2):
            s.targets[i] = np.clip(s.targets[i] + a[i, :3] * POS_SCALE, *WORKSPACE)
            s.rotations[i] = Rotation.from_rotvec(a[i, 3:6] * ROT_SCALE).as_matrix() @ s.rotations[i]
            s.grip_targets[i] = MAX_JAW * (1 - a[i, 6]) / 4
        was = s.success
        s.step()
        reward = float(s.success and not was)
        return self._obs(), reward, s.reason in ('success', 'bottle_fell', 'wrist_force_limit', 'numerical_instability'), s.reason == 'time_limit', self._info()

    def _obs(self):
        s = self.sim; d = s.data
        q = np.r_[d.qpos[s.qids[0]], d.qpos[s.qids[1]]]
        v = np.r_[d.qvel[s.dids[0]], d.qvel[s.dids[1]]]
        quat = [Rotation.from_matrix(d.site_xmat[t].reshape(3, 3)).as_quat(scalar_first=True) for t in s.tips]
        obs = {'joint_pos': q, 'joint_vel': v, 'gripper_qpos': d.qpos[np.ravel(s.finger_qids)].copy(),
               'eef_pos': d.site_xpos[s.tips].ravel().copy(), 'eef_quat': np.ravel(quat),
               'wrist_ft': np.concatenate([np.r_[d.sensor(x + '_force').data, d.sensor(x + '_torque').data] for x in SIDES]),
               'finger_force': s.finger_forces().ravel()}
        for c in self.cameras: obs[c] = s.render(c, self.image_size)
        return obs

    def _info(self):
        s = self.sim
        return {'success': s.success, 'termination_reason': s.reason, 'open_time': s.open_time, 'step': s.steps,
                'privileged': {'bottle_qpos': s.bottle_qpos(), 'bottle_pos': s.data.body('bottle').xpos.copy(), **s.mechanism()}}

    def render(self):
        return self.sim.render('agentview', 640)

    def close(self):
        self.sim.close()
