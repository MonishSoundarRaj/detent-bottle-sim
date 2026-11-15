"""Collect scripted-expert demonstrations at randomised bottle poses, in the Gym env's own
observation/action format. Only successful episodes are kept.

python -m bench.collect --n 100 --out data/expert_v1
python -m bench.collect --n 100 --noise .1 --out data/expert_noise10
python -m bench.collect --n 100 --settle --out data/expert_v2      # no idle waits (see bench/expert.py)

Each kept episode is data/<name>/episode_XXXX.npz with keys
  obs/<camera>        (T, S, S, 3) uint8
  obs/<proprio key>   (T, ...)      joint_pos, joint_vel, gripper_qpos, eef_pos, eef_quat, wrist_ft, finger_force
  action              (T, 14)       clean expert action in [-1, 1] (the label, even when noise was executed)
  target_pos          (T, 2, 3)     Cartesian tool target per arm before the action (for absolute-action policies)
  target_rotvec       (T, 2, 3)     tool target rotation per arm (axis-angle, world frame)
  bottle_qpos         (T, 10)       privileged object state (not for policy input)
  stage               (T,)          expert stage label
  pose                (3,)          bottle (x, y, yaw_deg)
plus manifest.jsonl (one line per attempt, kept or not) and config.json.
"""
import argparse
import json
import os
from multiprocessing import Pool
import numpy as np
from detent_bottle.gym_env import CAMERAS
from .poses import TRAIN_RANGE, split
from .rollout import expert_policy, rollout


def attempt(job):
    index, pose, args = job
    summary, episode = rollout(expert_policy(pose, args['settle']), pose, cameras=args['cameras'], image_size=args['image_size'],
                               record=True, noise=args['noise'], seed=index)
    summary['index'] = index
    if summary['success']:
        path = os.path.join(args['out'], f'episode_{index:04d}.npz')
        np.savez_compressed(path, pose=np.array(pose), **episode)
        summary['file'] = os.path.basename(path); summary['bytes'] = os.path.getsize(path)
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=100, help='number of training poses to attempt (first n of the train split)')
    p.add_argument('--start', type=int, default=0, help='skip the first poses of the split (to extend a dataset)')
    p.add_argument('--cameras', nargs='+', default=list(CAMERAS))
    p.add_argument('--image-size', type=int, default=96)
    p.add_argument('--noise', type=float, default=0., help='std of Gaussian noise on executed arm actions')
    p.add_argument('--settle', action='store_true', help='expert moves on once the robot is still instead of waiting out the clock')
    p.add_argument('--workers', type=int, default=max(1, os.cpu_count() - 1))
    p.add_argument('--out', required=True)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    config = dict(cameras=args.cameras, image_size=args.image_size, noise=args.noise, settle=args.settle, out=args.out)
    with open(os.path.join(args.out, 'config.json'), 'w') as f:
        json.dump(dict(config, train_range=TRAIN_RANGE.tolist(), start=args.start, n=args.n), f, indent=2)
    poses = split('train', args.start + args.n)[args.start:]
    kept = 0
    with Pool(args.workers) as pool, open(os.path.join(args.out, 'manifest.jsonl'), 'a') as f:
        for summary in pool.imap_unordered(attempt, [(args.start + i, pose, config) for i, pose in enumerate(poses)]):
            kept += summary['success']
            f.write(json.dumps(summary) + '\n'); f.flush()
            print(f"[{kept} kept] {summary}", flush=True)
    print(f'kept {kept}/{len(poses)} episodes in {args.out}')
