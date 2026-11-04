# docking_policies: Table III (closed-loop docking in simulation)

Closed-loop docking in SRB's zero-G lab task (`srb/zerog_lab_inspection`, `env.mode=docking`) on
the scenarios of the **docking test split** (benchmark_v3), for three kinds of policy:

| row | policy | input | where it runs |
|---|---|---|---|
| 9 | PPO teacher (privileged) | simulator state: robot state + goal relative to the true target | inside SRB (`srb agent eval`, the shipped policy) |
| 10 | BC students, one per input | RGB, depth, events, LiDAR, or all four (fused) + proprioception + docking command | `policy-server` (this package), behind SRB's policy bridge |
| 11 | FoundationPose → PPO | RGB-D → estimated target pose → the teacher's goal terms | `fp_oshd.closed_loop` (FoundationPose image), behind the bridge |

* **Upstream.** None. The code is written here and in SRB (`feat/inspection-datasets`):
  `srb/tasks/mobile/inspection/episode_log.py` and `policy_bridge.py`.
* **Weights.** The teacher is `osh_dataset_pipeline/sim/policies/zerog_lab_inspection/rsl_rl_ppo/20261015T114126/model_400.pt`, the policy that recorded the dataset. The students are trained here (`results/policies/bc/<input>/student.pt`).

## Protocol

**Scenarios** (`policy-scenarios`). Each docking test trajectory is one episode of an SRB scenario
process: 64 scenarios with 2–3 episodes each, 182 episodes in all. Every scenario is re-run with
its recorded Hydra overrides and seed. Its episodes run side by side, one environment each. This
gives the split's targets, mounts, scales, spins, lighting and docking speeds, none of which the
students saw in training.

**Start in view.** The recordings spawn the platform with a uniformly random heading, so
the target is often behind it. For the perception rows, every episode starts with the onboard
camera (base +x, 70° horizontal field of view) facing the target centre within ±10°
(`+env.spawn_face_target=true`, SRB `TaskCfg.spawn_face_target`). The heading's random draw is
reused as the offset, so the rest of the episode (target, spin, goals) is as recorded. The task
is then rendezvous and docking from a target in view, not a search.

**Physics and sensors.** `closed_loop.py --sensors` turns on the dataset recorder, writing into a
scratch directory that is deleted afterwards, so every row runs with the recordings' settings:

* 120 Hz physics and rendering
* RGB, depth and semantic at 20 Hz, all 320×240
* LiDAR at 10 Hz, 16 channels
* the event camera (v2e), only with `--events`, for the events and fused students; otherwise
  `event_camera_backend=noop`

The policies act at 10 Hz. Each episode lasts 40 s, as recorded.

**Docking command.** The goal's bearing `b` around the target centre and its distance `d` from it,
i.e. the task's pre-dock (1.2 m), hold, glide and docked schedule. The teacher receives the goal
relative to the true target; the students and FoundationPose → PPO receive `(cos b, sin b, d)` and
must find the target themselves. All policies observe the robot's own state: attitude, velocities,
IMU and previous action.

**Metrics** (`policy-metrics`, from the per-episode logs `episodes/*.npz`):

| column | definition |
|---|---|
| success | at the end of the episode the docking glide has finished (task phase 2), and the platform is within 5 cm (planar) and 5° (yaw) of the docked pose, moving at under 2 cm/s |
| pos. / att. err. | planar position and yaw error to the docked pose at the end, mean over episodes |
| contact vel. | planar speed relative to the target when the platform first reaches the docking standoff (centre distance ≤ standoff + 2 cm), mean over the episodes that reach it |
| thruster usage | Σ_t Σ_i u_i Δt, with u_i the applied thrust in [0, 1], in thruster-seconds |
| overlap (reported, not part of success) | whether the platform's body box ever intersects the target's CAD |

The simulated targets do not collide: arm-held targets have collisions disabled, and the docking
standoff (0.5–0.6 m) is a centre-to-centre distance that ignores the target's size. Of the
recorded docking runs, 40% end with the platform's box overlapping the target's CAD, mostly the
large ESA/NASA targets and the scaled CubeSats. The docking interface is therefore the standoff
pose, not a contact.

## Commands

```bash
cd baselines/docking_policies && uv sync
uv run policy-scenarios --out ../../results/policies/docking_test_v3/scenarios.json
cd ../..

# row 9: the teacher inside SRB
baselines/docking_policies/run_closed_loop.sh results/policies/docking_test_v3/scenarios.json \
    --out work/policies/docking_test_v3/teacher --workers 3 --sensors
uv run --project baselines/docking_policies policy-metrics work/policies/docking_test_v3/teacher

# row 10: BC students (cache of the docking train split, then one training per input)
uv run --project baselines/docking_policies policy-bc-cache --out work/bc_cache/docking_train_v3
uv run --project baselines/docking_policies policy-bc-train --cache work/bc_cache/docking_train_v3 \
    --input rgb --out results/policies/bc/rgb
uv run --project baselines/docking_policies policy-server --student results/policies/bc/rgb/student.pt --port 5555 &
baselines/docking_policies/run_closed_loop.sh results/policies/docking_test_v3/scenarios.json \
    --out work/policies/docking_test_v3/bc_rgb --workers 3 --sensors --server 127.0.0.1:5555
#   (--events for the events and fused students)

# row 11: FoundationPose -> PPO
FP_IMAGE=oshd-foundationpose:a1b694b-sm86 FP_DOCKER_ARGS="--network host" FP_NO_LOCK=1 \
    baselines/foundationpose/docker/run.sh python -m fp_oshd.closed_loop --port 5602 &
baselines/docking_policies/run_closed_loop.sh results/policies/docking_test_v3/scenarios.json \
    --out work/policies/docking_test_v3/fp_ppo --workers 2 --sensors --server 127.0.0.1:5602
```

`run_closed_loop.sh` runs `closed_loop.py` in the SRB container, with this repo, the pipeline and
SRB mounted, after `sim/install_policies.sh`. Finished scenarios are skipped, and a failed
simulator start is retried. `policy-server --policy teacher` runs the teacher behind the bridge,
and `--policy zero` sends no thrust; both are bridge checks.

## BC students

* **Data**: the released `steps`, `mocap`, `rgb`, `depth`, `lidar` and `events` streams of the
  docking train split (999 trajectories, 400k steps).
* **Inputs** (`inputs.py`, the same code in the cache and in the server):
  * RGB, depth and events at 128×96; events are per-polarity counts over the 50 ms before the step
  * LiDAR as a 16×360 range image
  * 26 proprioceptive terms: the teacher's observation without its goal terms
  * the docking command
* **Target**: the thrust the teacher computed from that step's observation. It is recorded at the
  next `steps` row (see `teacher.py`) and clipped to [0, 1] as applied.
* **Model** (`bc.py`): a small CNN per input with random shifts during training, and an MLP head
  ending in a sigmoid. MSE loss; 5% of the train trajectories are held out.
