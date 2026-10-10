import gymnasium as gym
import numpy as np
import detent_bottle  # noqa: F401


def test_reset_step_render():
    env = gym.make('DetentBottle-v0', image_size=64)
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    for name in ('agentview', 'frontview', 'topview', 'left_wrist', 'right_wrist'):
        assert obs[name].shape == (64, 64, 3) and obs[name].std() > 0
    assert 'bottle_qpos' in info['privileged'] and not any('bottle' in k for k in obs)
    q0 = info['privileged']['bottle_qpos'].copy()
    for _ in range(3):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    assert reward == 0. and not terminated and not truncated and info['step'] == 3
    assert np.isfinite(obs['joint_pos']).all() and np.abs(info['privileged']['bottle_qpos'] - q0).max() < .01
    assert env.render().shape == (640, 640, 3)
    env.close()


def test_reset_is_deterministic():
    env = gym.make('DetentBottle-v0', cameras=())
    env.reset(); a = np.full(14, .3)
    first = [env.step(a)[0]['joint_pos'] for _ in range(2)]
    env.reset()
    second = [env.step(a)[0]['joint_pos'] for _ in range(2)]
    assert np.array_equal(first, second)
    env.close()
