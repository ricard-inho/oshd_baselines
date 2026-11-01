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
3. **Same data, same scoring.** Use the frozen **`benchmark_v2`** (oshd 1.0.0, Hub revision
   `df3530b`), the committed sample lists in `results/sim/`, and `oshd-bench sim-eval` for the
   numbers. Never write your own metric code for a table cell.
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
cd bench && uv sync && cd ..                              # oshd 1.0.0 (TestPyPI) + oshd-bench
uv run --project bench oshd splits --benchmark v2         # train / test trajectory counts
hf auth login                                             # the dataset is public; login avoids rate limits
```

The dataset goes to `~/.cache/oshd/r3m3c3--oshd` (`$OSHD_DATA_DIR` moves it). Download only the
modalities you need, for one split at a time:

```bash
uv run --project bench oshd download --benchmark v2 --domain sim_zerog --task docking --split test \
    -m rgb -m semantic -m mocap            # + -m depth, -m events, -m lidar as needed
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

**Single-object methods (SPNv2).** The docking test split has 22 target models: satellite mockup
7467 samples, 18 procedural CubeSats 4081, three NASA models 689. A keypoint network is trained
for one object, so train and test SPNv2 on the satellite mockup only: build the training list
with `sim-prepare` and filter `object == 'satellite_mockup'`, and score with
`--subset "object == 'satellite_mockup'" --name <SETTING>_mockup`. FoundationPose on the same subset
scores e_t 3.65 cm, e_q 34.8°, S 0.690 (`baselines/foundationpose/results/sim_zerog_docking_test_v2/RESULTS_ORACLE_MASK_mockup.md`).

## Real data and the real ↔ twin transfer (Table II)

The real recordings (`real`, train 75 / test 17 trajectories; mockups `antena`,
`antena_1_solar`, plus 3 with the CubeSat) don't have a lab twin in the sim data yet: the sim
docking targets are other models. Shared real sample lists for the mockups, and the twin
recordings, are being prepared on `main` (ROADMAP rows #6–#8). Until they land, set up and
validate your method on the sim split; the transfer runs reuse the same code.

## When you are done

1. `RESULTS_<SETTING>.md`, the predictions and the metrics are committed under
   `baselines/<name>/results/<run>/`.
2. Your README has the full commands and timings.
3. Update your row in `ROADMAP.md` (status, the Paper row, the commit).
4. Open the pull request and send Ricard the Paper row.
