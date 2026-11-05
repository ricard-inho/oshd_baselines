# Experiment roadmap (paper Tables I–III)

This file tracks the experiments of the OSHD paper only, not the writing. Owners follow the
Notion plan. How to run one on your own branch: [`CONTRIBUTING.md`](CONTRIBUTING.md). Every experiment
uses the frozen **`benchmark_v2`** (oshd 1.0.0, Hub revision `df3530b`). Each row says what
must run, which code runs it, what blocks it, and roughly how long it takes on one RTX 3090.

Splits of `benchmark_v3` (trajectories; the first five rows are `benchmark_v2`'s):

| split | train | test |
|---|---:|---:|
| `sim_zerog/docking` | 921 | 162 |
| `sim_zerog/inspection` | 861 | 144 |
| `sim_3d/approach` | 1089 | 186 |
| `sim_3d/orbit` | 2528 | 522 |
| `real` (mockups `antena`, `antena_1_solar`, `cubesat`) | 75 | 17 |
| `twin` (lab twins of `antena`, `antena_1_solar`; in sim_zerog) | 160 | 40 |

Status legend: **done** · **running** · **ready** (code exists, only needs to run) ·
**adapt** (code exists for the real CubeSat and needs a sim variant) · **new** (no code yet) ·
**blocked**.

## Table I: pose estimation (`tables/tab_pose.tex`)

| # | Row | Owner | Status | Code | Data | GPU time | Notes |
|---|---|---|---|---|---|---|---|
| 1 | FoundationPose, RGB-D, docking, single-frame | Ricard | **done** (e_t 8.37 cm, e_q 43.0°, S 0.963) | `baselines/foundationpose` `fp_oshd.predict_sim` + `oshd-bench sim-eval` | `sim_zerog/docking/test`: 162 traj., 12 237 samples at 2 Hz | 4.9 h | oracle (simulator) masks; e_q and S are reported both strict and up to the measured symmetries |
| 2 | FoundationPose, RGB-D, inspection, tracking | Ricard | **done** (e_t 14.76 cm, e_q 13.1°, S 0.348; coverage 0.84) | add `--mode track` to `predict_sim`: `register()` on the first frame, then `track_one()` at 20 Hz, re-register on loss; add drift-vs-time to `eval_sim` | `sim_zerog/inspection/test`: 144 traj. (download rgb, depth, semantic, mocap) | ~4 h (tracking is much faster than `register()`) | metric: e_t / e_q / S over time since initialisation, plus mean over the run |
| 3 | SPNv2, RGB, docking | Aman | **done** (e_t 2.13 cm, e_q 5.8°, S 0.125; satellite-mockup subset) | `baselines/spnv2` `spnv2-list --exp sim` + `oshd-bench sim-eval --subset "object == 'satellite_mockup'"` | train: the 384 `satellite_mockup` runs of `sim_zerog/docking/train` (57 789 samples at 4 Hz); test: its 99 runs in `.../test` (7 467 of the 12 237 samples) | 14.2 h train + 3 min test (RTX 4070) | one network for the satellite mockup (SPNv2 regresses the keypoints of one object); RGB only, no mask at test time. FoundationPose on the same subset: e_t 3.65 cm, e_q 34.8°, S 0.690 |
| 4 | SPNv2, events (time surfaces), docking | Matteo | **adapt** | same as #3, with a time-surface input (`events` of the sim runs, 50 ms windows) | as #3 + `-m events` | ~1 day | skip if it does not train without tuning (Notion) |
| 5 | LiDAR ICP to CAD, docking, single-scan | Ricard | **done** (e_t 8.12 cm, e_q 19.7°, S 0.568) | `baselines/lidar_registration` (FPFH + TEASER++ + ICP, training-free) on sim scans; CAD from `models/sim` | `sim_zerog/docking/test` `-m lidar -m mocap` | CPU, ~2 h | the sim LiDAR is 10 Hz (300 scans per 30 s run) |

## Table II: domain transfer real ↔ lab twin (`tables/tab_transfer.tex`)

| # | Row | Owner | Status | Notes |
|---|---|---|---|---|
| 6 | SPNv2 trained on real, tested on twin | Aman (RGB), Lara (events) | RGB: **done** (e_t 48.75 cm, e_q 110.3°, S 2.428); events: **ready** | RGB: `baselines/spnv2` `spnv2-list --exp real` (one network per mockup) on `results/real/real_realsense_color_train_v3` (64 traj.), tested on `results/sim/twin_test_v3` (40 traj., 3038 samples) with `oshd-bench sim-eval`; 6.0 h train + 1 min test (RTX 4070). No transfer (chance-level rotation). In-domain reference on the real test split: e_t 21.71 cm, e_q 46.2°, S 0.944 (fits its training samples to 1.8 cm / 1.7°: few distinct real views). Unsupervised adaptation on unlabelled twin-train images (variant, not the paper row): AdaBN 44.83 cm / 112.1° / 2.532, ODR 44.10 cm / 110.7° / 2.488. See CONTRIBUTING.md, "Real data and the real ↔ twin transfer" |
| 7 | SPNv2 trained on twin, tested on real | Aman (RGB), Lara (events) | RGB: **done** (e_t 124.28 cm, e_q 110.1°, S 2.673); events: **ready** | RGB: `baselines/spnv2` `spnv2-list --exp twin` on the twin train split (`sim-prepare --benchmark v3 --domain twin --task all --split train --stride 5`: 160 traj., 24 332 samples), tested on `results/real/real_realsense_color_test_v3` (16 traj., 14 594 samples) with `oshd-bench sim-eval`; 6.0 h train + 6 min test (RTX 4070). No transfer: the network reads the dark real frames as a huge, close target. In-domain reference on the twin test split: e_t 4.75 cm, e_q 10.1°, S 0.236. Unsupervised adaptation on unlabelled real-train images (variant, not the paper row): AdaBN 116.70 cm / 104.3° / 2.529, ODR 119.24 cm / 100.7° / 2.479 |
| 8 | FoundationPose zero-shot on both domains | Ricard | **twin: ready; real: blocked on the depth registration** | real RGB-D needs the `realsense_depth` → colour registration (DATA_REQUIREMENTS items 1 and 8: missing / verify), and a 2D mask on real frames (the Mask R-CNN in `baselines/detector`, or GT-rendered masks as an oracle). Twin side: as #1 on the twin runs. |

## Table III: visuomotor policies (`tables/tab_policy.tex`)

| # | Row | Owner | Status | Notes |
|---|---|---|---|---|
| 9 | PPO teacher (privileged), closed-loop docking in sim | Ricard | **adapt** | the policy that drove the docking recordings ships in `osh_dataset_pipeline/sim/policies/zerog_lab_inspection` (`env.mode=docking`). Needs an SRB eval script that logs success, final position/attitude error, contact velocity and thruster usage over N seeded episodes. Define success (paper TODO). |
| 10 | BC students, one per modality (RGB, depth, events, LiDAR, fused) | Ricard | **new** | train from `steps` (observations, actions) + the sensor streams of `sim_zerog/docking/train`; evaluate closed loop in SRB with the #9 harness. 5 trainings. |
| 11 | Modular: FoundationPose → PPO teacher, closed loop | Ricard | **new** | needs FoundationPose in the SRB loop: run it as a service (the Docker image of #1) that SRB queries per policy step, or use tracking mode (#2) for speed |

## Order

1. #1 (done), then publish oshd 1.0.0 to PyPI.
2. #2 and #5: no training, they reuse #1's infrastructure.
3. #3 / #4: decide on the single-target subset first.
4. #9, then #10 and #11 (they share the closed-loop harness).
5. #6–#8, after the twin question and the depth registration are resolved.

## Reproducing a finished row

Each baseline's `README.md` has the exact commands. The shared sample lists and ground truth of
the sim splits are in `results/sim/<run>/` (`oshd-bench sim-prepare`); each baseline's
predictions, metrics (`oshd-bench sim-eval`) and a few overlays are committed under
`baselines/<name>/results/<run>/`.
