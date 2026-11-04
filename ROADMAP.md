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
| 3 | SPNv2, RGB, docking | Aman | **adapt** | `baselines/spnv2` is written for the real CubeSat_U1 | train on `sim_zerog/docking/train`, test on `.../test` | ~1 day train + 1 h test | SPNv2 regresses keypoints of **one** object, but the docking split has 22 targets. Proposal: train and test on the `satellite_mockup` runs only (7 467 of the 12 237 test samples), and report FoundationPose on the same subset for the comparison |
| 4 | SPNv2, events (time surfaces), docking | Matteo | **done** (e_t 6.73 cm, e_q 17.0°, S 0.387; `satellite_mockup` subset, 7 467 samples; `TS_canon` ablation 6.55 cm, 15.2°, 0.359) | `baselines/spnv2_events` (`spnv2ev-train` / `spnv2ev-predict` + `oshd-bench sim-eval --subset "object == 'satellite_mockup'" --name TS_mockup`) | 384 mockup train runs (`-m rgb -m semantic -m mocap -m events`, 40 GB) + docking test `-m events -m mocap` (12 GB) | 6.9 h train (RTX 4070), 4 min test | 3-channel time surface of the 50 ms window (τ 25 ms), 8 bbox-corner keypoints × run `scale`, labels canonicalised over the 3 measured flips + symmetry-aware (min over flips) heatmap/pose losses, checkpoint picked on 32 held-out train runs; upstream's recipe, no tuning. FoundationPose on the same subset: e_t 3.65 cm, e_q 34.8°, S 0.690 |
| 5 | LiDAR ICP to CAD, docking, single-scan | Ricard | **done** (e_t 8.12 cm, e_q 19.7°, S 0.568) | `baselines/lidar_registration` (FPFH + TEASER++ + ICP, training-free) on sim scans; CAD from `models/sim` | `sim_zerog/docking/test` `-m lidar -m mocap` | CPU, ~2 h | the sim LiDAR is 10 Hz (300 scans per 30 s run) |

## Table II: domain transfer real ↔ lab twin (`tables/tab_transfer.tex`)

| # | Row | Owner | Status | Notes |
|---|---|---|---|---|
| 6 | SPNv2 trained on real, tested on twin | Aman (RGB), Lara (events) | **ready** (twins in benchmark_v3; sample lists in `results/real/`, `results/sim/twin_test_v3/`) | train on the real train split (64 traj., mockups), test on the 40 twin test trajectories; see CONTRIBUTING.md, "Real data and the real ↔ twin transfer" |
| 7 | SPNv2 trained on twin, tested on real | Aman (RGB), Lara (events) | **ready** | as #6 |
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
