# bench: demos, shared evaluation and an ACT baseline

Everything here runs from the repo root and treats `DetentBottle-v0` exactly as shipped. Physics,
cameras, action and observation spaces are unchanged.

## Why

`reset(seed)` always gives the same state and the reference opener uses fixed world coordinates.
So imitation data from it is a single trajectory, and a policy tested at the default pose could
just be memorising it. `bench` varies the bottle pose (`bottle_pose=(x, y, yaw_deg)`) for both
training and testing.

## Findings so far

- **Env bug, fixed in `detent_bottle/scene.py`:** `bottle_pose` yaw is documented in degrees,
  but the scene compiles with `angle="radian"`. So `yaw_deg=10` rotated the bottle by -147 deg.
  It now converts degrees to radians. `tests/test_bench.py::test_bottle_pose_yaw_is_degrees`
  covers it.
- **Expert reach (one axis at a time, `bench/sweep.py`):** it succeeds at x and y offsets of
  +-2 to +-8 cm, except x = -4 cm, where it folds the loop but misses the button. For yaw it
  succeeds at -10 and +10 to +45 deg, and fails at -20, -30 and -45 deg (the loop is never
  folded). These poses are kinematically reachable, so the failure is in contact.
  The expert is not perfect even close to the default pose. `collect.py` keeps only successful
  runs, and the expert's own success on each test split is the reference score.
  `data/expert_v1` holds 100 successful demos from 106 train poses (590 MB at 96 px). Every
  failure folded the loop but missed the button press. Five of the six were at |y| of 3 to 4 cm
  with x near 0.
- **Pose ranges (`bench/poses.py`):** train and test_id use x, y +-4 cm and yaw +-10 deg.
  test_ood goes up to +-6 cm and +-20 deg. The fixed close-up cameras still see the latch
  across this range.
- **The env does not detect a tipped bottle:** `bottle_fell` needs z < .76 m, which only happens
  if the bottle leaves the table. A bottle lying on its side runs to the time limit. The harness
  (`bench/rollout.py`) ends those episodes as `bottle_tipped` once the tilt passes 60 deg. The
  env itself is unchanged.
- **The reference opener's idle waits make its demos unlearnable for a reactive policy.** Every
  segment lasts a fixed number of ticks. In most of them the arm reaches its target early, then
  sits still (25-46 ticks, joint motion exactly 0, camera frames nearly identical) until the
  clock runs out, and only then does the gripper close or the next move start. A policy that
  sees only the current observation cannot tell when the wait is over. ACT trained on 100 such
  demos (`expert_v1`) reaches the bottle within a few mm but never closes either gripper, so it
  scores 0/50. In open-loop replay on held-out demos its targets are within 1-3 mm at the grip
  and pinch, so the model and the deployment code are not the problem. Diffusion policy or a VLA
  trained on the same demos will hit the same wall. `ScriptedExpert(settle=True)`
  (`collect.py --settle`) ends a segment once both tools are on target and the robot has been
  still for 3 ticks, keeping the original tick count as the upper bound. Episodes drop from
  ~714 to ~524 ticks. Contact segments (pinch, fold, press) still run their full length. The
  default `settle=False` reproduces `expert_v1` action for action.
- **Results so far (`python -m bench.report`):**

  | policy | split | success | loop folded | notes |
  |---|---|---|---|---|
  | scripted expert (privileged pose) | test_id (30) | 96.7% | 100% | 1 missed button press |
  | scripted expert (privileged pose) | test_ood (20) | 85% | 90% | fails at yaw near -15 deg, plus missed presses |
  | settle expert (privileged pose) | test_id (30) | 100% | 100% | ~520 ticks per episode instead of ~715 |
  | settle expert (privileged pose) | test_ood (20) | 90% | 90% | both failures at yaw -14.5 and -17 deg |
  | ACT quick pass (53 demos, frozen backbone, layer 3, 5k steps) | test_id (first 20, 1200-tick cap) | 0% | 5% | Copies the overall motion but tips the bottle while gripping it (9/20) |
  | ACT on `expert_v1` (100 timed demos, layer 3, 20k steps) | test_id (30) / test_ood (20) | 0% / 0% | 0% / 0% | Never closes either gripper (the idle-wait problem above) |
  | **ACT on `expert_v2`** (100 settle demos, layer 3, 20k steps) | test_id (30) | **86.7%** | 86.7% | ~500 ticks per success. 4 failures: 2 at x=+3.7 cm, y=+2 cm never reach the loop; 1 folds the loop but misses the button; 1 tips the bottle mid-fold |
  | **ACT on `expert_v2`** | test_ood (20) | **35%** | 45% | Mostly fails outside the training range, usually before folding the loop, often after pushing the bottle 4-7 cm |

- **Training speed (MPS), 5 cameras at 96 px, batch 32:** on an M5 (10-core GPU), 1.6 steps/s
  with the ResNet trained, 4 steps/s with `--freeze-backbone`, and 0.82 steps/s with the ResNet
  trained and `--feature-layer 3`. On an M5 Pro (16-core GPU), 3.2 steps/s with the ResNet
  trained and `--feature-layer 3`, so 20k steps take about 1.7 h. bf16 autocast and
  channels-last did not help.

## Pieces

| module | what it does |
|---|---|
| `bench/expert.py` | The reference opener moved rigidly into the bottle's frame and driven through the Gym action space. It is privileged (told the pose) |
| `bench/actions.py` | Delta action ↔ absolute tool-target conversion. Mirrors `DetentBottleEnv.step` exactly |
| `bench/poses.py` | Pose ranges and the fixed `train` / `test_id` / `test_ood` splits |
| `bench/rollout.py` | Runs one episode and summarises it. `python -m bench.rollout x y yaw` runs the expert |
| `bench/sweep.py` | Runs the expert over many poses in parallel (reachability check) |
| `bench/collect.py` | Parallel demo collection in Gym format. Keeps only successful episodes |
| `bench/evaluate.py` | Shared evaluation harness for any policy |
| `bench/act/` | ACT baseline: model, in-memory dataset, trainer, policy wrapper |

## Collect demonstrations

Use `--settle` for any policy that learns from these demos (see Findings). `data/expert_v2` is 100
settle demos (100/100 kept, 523 MB).

```bash
python -m bench.collect --n 100 --settle --out data/expert_v2         # demos without idle waits (recommended)
python -m bench.collect --n 100 --out data/expert_v1                  # timed demos, as the reference opener runs
python -m bench.collect --n 100 --noise .1 --out data/expert_noise10  # noisy execution, clean labels
```

Each `episode_XXXX.npz` holds `obs/<key>` for every Gym observation key, `action` (T, 14), the
env's Cartesian targets before each action (`target_pos`, `target_rotvec`), privileged
`bottle_qpos`, the expert `stage` label and the `pose`. The default is 96 px images from all
five cameras. That is about 6 MB per episode, because zlib can reuse the previous frame at this
size. At 128 px it is about 34 MB.

With `--noise`, the executed arm motion gets Gaussian noise but the recorded `action` is the
clean expert action (DART-style). This gives recovery data.

## Evaluate a policy

Any policy is a factory `make_policy(pose, **kwargs)` that returns `policy(obs, sim) -> action`.
Learned policies must ignore `pose` and `sim`. They may set `cameras`, `image_size` and
`reset(sim)` attributes.

```bash
python -m bench.evaluate --policy bench.rollout:expert_policy --split test_id test_ood --name expert
python -m bench.evaluate --policy bench.act.policy:make_policy --kwargs '{"ckpt": "runs/act/v1/last.pt"}' --split test_id test_ood --name act_v1
```

Results go to `runs/eval/<name>/` (`<split>.jsonl` per episode and `summary.json`). Reported:
success rate, how often the loop was folded past the detent (>150 deg), how often the lid opened
(>60 deg), mean steps for successful episodes, and termination reasons.

## Train ACT

```bash
python -m bench.act.train --data data/expert_v1 --out runs/act/v1 --steps 20000
python -m bench.act.train --data data/expert_v1 --out runs/act/v1 --steps 20000 --resume  # after an interruption
```

`last.pt` is written every 1000 steps with the model, optimiser state and step, so `--resume`
loses at most 1000 steps. Pass the same arguments as the original run. On a laptop, wrap the
command in `caffeinate -i` so macOS does not sleep mid-run.

The model predicts 50-tick chunks of absolute tool targets per arm (position, 6-D rotation,
jaw). At run time it tracks the env's targets itself from the known reset values, ensembles
overlapping chunks (ACT temporal ensembling) and converts the result to a delta action.
