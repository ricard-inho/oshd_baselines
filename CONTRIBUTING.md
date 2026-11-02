# Running a baseline on your own branch

This is for everyone who runs one of the paper's experiments (rows of [`ROADMAP.md`](ROADMAP.md)).
It is written so that it can be handed to Claude Code as is: "read CONTRIBUTING.md and
ROADMAP.md row #N, then set up and run it".

## Rules

1. **Branch.** Work on `exp/<your-name>-<short-name>` from `main`, for example
   `exp/aman-spnv2-rgb`. Open a pull request to `main` when your numbers are final. Ricard merges.
2. **Stay in your folder.** Change only `baselines/<your-baseline>/`. If you need a change in
   `bench/` (the shared sample lists, formats and metrics) or in another baseline, make it a
   separate small pull request first, so the experiment branches don't conflict.
3. **Same data, same scoring.** Use the frozen **`benchmark_v3`** (oshd 1.1.0, Hub revision
   `c8e5ca0`; it contains `benchmark_v2` with the same splits), the committed sample lists in
   `results/sim/` and `results/real/`, and `oshd-bench sim-eval` for the numbers. Never write
   your own metric code for a table cell.
4. **No test labels at inference.** Inference reads `samples.parquet` only. `gt.parquet` of a
   **test** split is read by `oshd-bench sim-eval` and nothing else. Training code may read the
   train split's `gt.parquet`.
5. **Don't edit the paper.** Send Ricard the "Paper row" line of your `RESULTS_*.md` and the
   commit hash; the tables are filled from there.
6. **Reproducible.** Your baseline's `README.md` must let someone else get your numbers:
   upstream repo + pinned commit, licence, where the weights come from, the environment (a uv
   project, or a Docker image), and the exact commands from download to `sim-eval`, with how long
   each step took on which GPU. `baselines/foundationpose/README.md` is the model to follow.
7. **Big files stay out of git:** `third_party/`, `weights/`, `work/`, `*.log` are ignored.
   Commit the predictions parquet, the metrics, `RESULTS_*.md` and a handful of overlays.

## Setup (once)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh          # uv
git clone https://github.com/ricard-inho/oshd_baselines.git && cd oshd_baselines
git switch -c exp/<your-name>-<short-name>
cd bench && uv sync && cd ..                              # oshd 1.1.0 (release tag v1.1.0) + oshd-bench
uv run --project bench oshd splits --benchmark v3         # train / test trajectory counts
hf auth login                                             # the dataset is public; login avoids rate limits
```

The dataset goes to `~/.cache/oshd/r3m3c3--oshd` (`$OSHD_DATA_DIR` moves it). Download only the
modalities you need, for one split at a time:

```bash
uv run --project bench oshd download --benchmark v3 --domain sim_zerog --task docking --split test \
    -m rgb -m semantic -m mocap            # + -m depth, -m events, -m lidar as needed
uv run --project bench oshd download --benchmark v3 --domain twin -m rgb -m semantic -m mocap    # lab twins
uv run --project bench oshd download --benchmark v3 --domain real -m realsense_color -m mocap    # real
```

The docking test split with rgb, depth, semantic and mocap is about 12 GB; the train split is about
6x that. The sim events of a run are about as large as its rgb frames (~70 MB per run).

On the shared GPU machine, run GPU jobs under `flock work/gpu.lock <command>` so they don't run
at the same time as someone else's.

## Sim splits: sample lists, predictions, evaluation

The sample list of a sim split is shared by every method:

| path | contents |
|---|---|
| `results/sim/sim_zerog_docking_test_v2/samples.parquet` | `trajectory`, `index` (rgb frame index), `t_ns`, `object` (target model), `scale`, `mask_px`, `mount`: every 10th rgb frame (2 Hz) where the target's mask has ≥ 100 px. 12 237 samples, 162 trajectories |
| `.../gt.parquet` | + `T` (`T_cam_target`, 16 floats row-major) and `range_m` |
| `.../objects.json` | target model → CAD `models/sim/<model>/model.obj` (relative to the dataset root; metres, **unscaled**: multiply by the sample's `scale`), extent, kind |
| `.../prepare.json` | settings and Hub revision |

`T_cam_target` is the pose of the target's root frame in the optical frame of the sim `rgb`
camera (x right, y down, z forward), in metres. The sim event camera is pixel-aligned with `rgb`
(same intrinsics and pose), so event methods use the same samples and the same `T`, with the
events of a window ending at `t_ns`. Read frames, intrinsics, events and poses through oshd
(`OshDataset(benchmark="v2")[trajectory]`; `traj.frame("rgb", index)`,
`traj.calibration.intrinsics("rgb").K`, `traj.target_mesh()`; see the oshd README).

Training sample lists are not committed. Build the one you need with the same command:

```bash
uv run --project bench oshd-bench sim-prepare --domain sim_zerog --task docking --split train
#  -> results/sim/sim_zerog_docking_train_v2/ (gt.parquet has the labels for training)
```

**Predictions.** Write them with `oshd_bench.PoseWriter` to
`baselines/<name>/results/sim_zerog_docking_test_v2/predictions_<SETTING>.parquet`. The writer
also writes a `.json` with your metadata: put the method, the upstream commit, the checkpoint and
the input in it.

```python
from oshd_bench import PoseWriter
w = PoseWriter(path, {"method": "SPNv2", "commit": "...", "checkpoint": "...", "input": "sim rgb"})
w.add(trajectory, "rgb", t_ns, object_name, T_cam_target_4x4, score, runtime_ms)
w.save()
```

A sample without a prediction counts as a failure. Don't drop the hard ones.

**Evaluation**:

```bash
uv run --project bench oshd-bench sim-eval --run-dir results/sim/sim_zerog_docking_test_v2 \
    --pred baselines/<name>/results/sim_zerog_docking_test_v2/predictions_<SETTING>.parquet \
    [--subset "object == 'satellite_mockup'" --name <SETTING>_mockup]
```

It writes `per_sample_*`, `metrics_*.json|csv` and `RESULTS_<SETTING>.md` next to the predictions.
The first lines of `RESULTS_*.md` contain the **Paper row**: means over the samples of e_t [cm],
e_q [°] and the SPEED score S, with the rotation error taken up to the target's measured
symmetries. The per-range and per-target tables below it are what the discussion uses.

**Single-object methods (SPNv2), Table I.** Table I uses the `benchmark_v2` docking test split
(`results/sim/sim_zerog_docking_test_v2`: the same runs as v3's docking test without the twins).
It has 22 target models: satellite mockup
7467 samples, 18 procedural CubeSats 4081, three NASA models 689. A keypoint network is trained
for one object, so train and test SPNv2 on the satellite mockup only: build the training list
with `sim-prepare` and filter `object == 'satellite_mockup'`, and score with
`--subset "object == 'satellite_mockup'" --name <SETTING>_mockup`. FoundationPose on the same subset
scores e_t 3.65 cm, e_q 34.8°, S 0.690 (`baselines/foundationpose/results/sim_zerog_docking_test_v2/RESULTS_ORACLE_MASK_mockup.md`).

## Real data and the real ↔ twin transfer (Table II)

`benchmark_v3` has the **lab twins** of the real mockup recordings: 200 sim_zerog trajectories
(100 per mockup) with the real mockups' CAD standing at the poses measured in the real
recordings, flying the same inspection and docking tasks. oshd selects them with
`domain="twin"`; the real ones with `domain="real"`.

| | train | test | shared sample list (test) |
|---|---:|---:|---|
| real (mockups `antena`, `antena_1_solar`) | 64 traj. / 49 382 frames | 16 traj. / 14 594 frames | `results/real/real_realsense_color_test_v3/` |
| twin (`real_antena`, `real_antena_1_solar`) | 160 traj. | 40 traj. / 3 038 samples | `results/sim/twin_test_v3/` |

* **Real → Twin:** train on the real train split, predict on `results/sim/twin_test_v3`.
* **Twin → Real:** train on the twin train split, predict on `results/real/real_realsense_color_test_v3`.
* Training lists (with labels): `oshd-bench real-prepare --split train`
  (`results/real/real_realsense_color_train_v3/`, committed) and
  `oshd-bench sim-prepare --benchmark v3 --domain twin --task all --split train` (build it).
* **Frames.** Real: the RealSense colour camera (`realsense_color`, 640x480 with distortion:
  undistort with `traj.calibration.cameras["realsense_color"]`), `T` = `T_camera_model` in the
  mockup's CAD frame (`models/real/<mockup>/model.obj`). Twin: the sim `rgb` camera (320x240,
  pinhole), `T` = `T_camera_target` in the same CAD frame (the twin's `models/sim/real_<mockup>/`
  is the real CAD). So one keypoint set per mockup serves both domains.
* **Events.** Real: the EVK4 (`events`, calibrated from trajectory_010 on; window ending at the
  frame time). Twin: the sim event camera, pixel-aligned with `rgb`.
* Score both directions with `oshd-bench sim-eval --run-dir <test list> --pred <predictions>`. The
  real ground truth was checked by projecting the CAD into the RealSense images; the CubeSat
  recordings (007-009) are left out because their ground truth is off.
* A single-object network (SPNv2) trains one model per mockup, or one with both mockups' keypoints:
  your call; say which in your README.

## When you are done

1. `RESULTS_<SETTING>.md`, the predictions and the metrics are committed under
   `baselines/<name>/results/<run>/`.
2. Your README has the full commands and timings.
3. Update your row in `ROADMAP.md` (status, the Paper row, the commit).
4. Open the pull request and send Ricard the Paper row.
