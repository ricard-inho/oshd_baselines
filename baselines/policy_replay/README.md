# policy_replay: Table II

The frozen PPO policy (Pingu, GoThroughPoses) is replayed on the recorded runs. At each 10 Hz
policy step, the privileged mocap-derived robot state in its observation is replaced by the
state derived from a pose estimator. The Table II metric is ‖a_replay(estimator) − a_replay(mocap)‖₂
in the normalised action space (the `delta_a_vs_mocap_replay` column), so the mocap row is 0 by
construction and observation-rebuild error cancels out. Each row reports its mean and P95. The
rebuild check, ‖a_replay(mocap) − a_logged‖₂ (the `delta_a` column of `mocap.parquet`), is
reported once next to the table (`mean_delta_a_vs_logged` in `table2_replay.csv`).

* **Upstream.** None. The code is written here.
* **Weights.** `exported/policy.onnx` from `pooh_training/2026-09-22_17-01-14_ppo_Pingu_GoThroughPoses_rsl_rl_seed_42`. It is identical to `model_7999.pt`. Licence: that of the training repo (rans_v2_priv, private).
* **Environment.** A uv project that runs onnxruntime on the CPU.

## Commands

```bash
cd baselines/policy_replay && ./setup.sh
uv run policy-replay verify                  # mocap reconstruction vs logged actions (all trajectories)
uv run policy-replay all                     # mocap row + best (lowest S_real) method per group,
                                             # read from results/tables/table1_pose.csv
uv run policy-replay all --source rgb=rgb_firefly/spnv2/FT   # pin a row
uv run policy-replay run --key lidar --source lidar/geotransformer/FT [--latency]
cd ../.. && uv run --project bench pooh-bench eval           # fills table2_replay.csv/.tex
```

Outputs:

* `results/replay/<key>.parquet`, one row per policy step, with these columns:
  * `trajectory`, `t_ns`, `delta_a`
  * `delta_a_vs_mocap_replay`
  * per-component `d_<action>`
  * `a_replay_*`, `a_logged_*`
  * `has_estimate`, `estimate_age_ms`
* `<key>.json`, which holds:
  * `estimator` and `source`
  * statistics per trajectory
  * the number of steps without any estimate yet (their `delta_a` is NaN)
  * the full config

The policy runs on the `test` split of the real domain (`--split`, `--domain`).

Smoke test on the dev dataset, with **fake** estimators (GT + noise + a random cube symmetry):

```bash
export POOH_DATASET_ROOT=$PWD/work/dev_dataset POOH_BENCH_RESULTS=$PWD/work/dev_results
uv run policy-replay fake-predictions --modality rgb_realsense --symmetric   # method FAKE_gt_noise
uv run policy-replay run --key rgb --source rgb_realsense/FAKE_gt_noise/ZS
```

## Observation reconstruction

The layout lives in `configs/replay.yaml` and is declarative: 26 named features in order, each with an optional scale. The first 13 entries come from `GoThroughPosesTask.get_observations`:

* v_body x, v_body y and ω_z
* distance to the goal
* cos and sin of (bearing to goal − heading)
* cos and sin of (goal heading − heading)
* the 5 terms of the "subsequent goal"

The last 13 entries are Pingu's `_previous_actions`: 8 thrusters, 4 arm joints and the reaction wheel.

**Logged actions.** They come from `/autonomous_control_input`, one row per policy step where flags 0 and 1 are both 1:

* **Thrusters** are in [0, 1], and a = 2u − 1.
* **Arms** are always 0, because the arms are fixed on the robot. They are not compared, and their previous-action entries are 0.
* **Reaction wheel.** The logged value is the wheel-model reaction torque from `pingu.py`. It is inverted to recover a, with ω = 0 at the first step.

So `delta_a` covers 9 of the 13 dimensions.

**Estimator substitution.** Each estimate T_sensor_model is resolved over the 24 cube symmetries using the gravity direction. The base pose is then built as:

T_world_base = T_world_cube(mocap median) · inv(T_sensor_model) · inv(T_SensorStack_sensor) · T_SensorStack_floatingplatform(mocap median)

The state at each step then comes from two rules:

* **Pose.** The latest estimate at or before the step is held (zero-order hold). With `--latency`, an estimate only counts from t + runtime_ms.
* **Velocity.** `velocity.mode: buffer_mean` copies the robot's `OptitrackStatePreProcessor` (RANS_DeployToRobot): the mean finite difference over the last 30 pose samples, divided by (t_last − t_first)/30. `mode: fit` is a causal least-squares line over the last `window_ms`. The mocap row and every estimator use the same filter.

## Status: the mocap reconstruction does NOT reproduce the logged actions

**2026-09-29.** The policy is confirmed (the ONNX above). The deployment code is `~/UniLuFP/ros_ws/src/RANS_DeployToRobot` (`rl_inference`, `rl_task_node_v2`), but the local checkout (`433b963`, 2026-09-10) predates the policy. None of its formatters produce this 26-dim GoThroughPoses layout: `GoToPoseFormater` gives `[dist, bearing cos/sin, heading cos/sin, vx, vy, ω] + last actions` = 21, and `pingu.yaml` has `cubo_mode: true`. The code that ran is still needed, from the robot's onboard computer, together with its `ros_experiments_logs` CSVs.

From that checkout, the wheel command is `rw = a[-1] · 0.05` (so `rw_mode: linear`), and the velocity filter is `buffer_mean`. Neither change closes the gap: mean ‖Δa‖ is 1.24 / 1.33 / 1.41 (thrusters alone ≈ 0.85–1.07). So the mismatch lies in the observation layout, not in velocity or wheel scaling. The table below is the earlier state (fit velocity, wheel model).

| trajectory | mean ‖Δa‖ | median | P95 | per-dimension correlation |
|---|---|---|---|---|
| 007 | 1.10 | 0.81 | 2.06 | thrusters 0.66–0.93, wheel 0.56 |
| 008 (base pose via SensorStack) | 1.36 | 1.40 | 1.99 | 0.85–0.94, wheel 0.69 |
| 009 | 1.41 | 1.44 | 2.08 | 0.83–0.94, wheel 0.53 |

The thruster outputs track the logged ones: the correlation matrix is clearly diagonal, so the ordering and signs are right. They are not numerically equal, though. None of the following brings the mean below about 0.9:

* the prev-action lag (1, 2 or 3)
* the velocity filter (finite differences over 50–200 ms, regression over 150–500 ms)
* the obs/action time shift (±3 steps)
* a heading offset (0, ±90°, 180°) or a mirrored world
* a base-origin offset
* the subsequent-goal fields: self-loop, zeros, origin, or fitted freely
* arm prev-actions: 0, self-fed, raw or clipped
* the wheel mapping
* the other checkpoints, `model_0`…`model_7999`

Free fits of 3–7 parameters converge to inconsistent values. The largest errors are in the approach phase, where the state is least ambiguous. The recorded runs were most likely driven by another policy, or by a different observation formatter.

The `delta_a_vs_mocap_replay` column measures the effect of the estimator alone: ‖a_replay(estimator) − a_replay(mocap)‖. With exact GT "estimates" at RealSense rate, it is 0.55 (window 300 ms) and 0.20 (window 1000 ms). That residual comes from:

* the lower pose rate and the zero-order hold
* the cube and rig medians
* the policy's bang-bang sensitivity near the goal
