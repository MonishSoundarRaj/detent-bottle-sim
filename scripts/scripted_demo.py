"""Run the scripted reference opener on the detent-latch bottle and report success.

Optionally saves an agentview video (--video out.mp4, needs imageio[ffmpeg]) or an
.npz of per-tick qpos/targets for imitation data (--save demo.npz)."""
import argparse
import time
import numpy as np
from detent_bottle import DetentBottleSim
from detent_bottle.policy import opening_sequence

p = argparse.ArgumentParser()
p.add_argument('--seed', type=int, default=0, help='accepted for API symmetry; the task and opener are deterministic')
p.add_argument('--save'); p.add_argument('--video')
args = p.parse_args()
np.random.seed(args.seed)
t = time.time()
e = DetentBottleSim(stop_on_success=True)
rows, frames = [], []
for stage, arm, pos, rpy, gap, ticks in opening_sequence():
    e.target(arm, pos, rpy, gap)
    for _ in range(ticks):
        e.step()
        rows.append(dict(qpos=e.data.qpos.copy(), targets=e.targets.copy(), rotations=e.rotations.copy(), grips=e.grip_targets.copy()))
        if args.video and e.steps % 2 == 0: frames.append(e.render('agentview', 480))
        if e.reason: break
    print(f'{e.steps:4d} {stage:50s} {e.mechanism()}', flush=True)
    if e.reason: break
if args.save:
    np.savez_compressed(args.save, **{k: np.array([r[k] for r in rows]) for k in rows[0]})
if args.video:
    import imageio.v2 as imageio
    imageio.mimsave(args.video, frames, fps=10)
print(f'latch=detent success={e.success} reason={e.reason} steps={e.steps} sim_s={e.data.time:.2f} wall_s={time.time() - t:.1f}')
e.close()
