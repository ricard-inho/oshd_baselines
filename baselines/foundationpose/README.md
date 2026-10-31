# FoundationPose (RGB-D) and FoundationPose + ICP

Table I, block **RGB-D (RealSense)**: modality `rgbd_realsense`, sensor `realsense_color`, rows
`foundationpose` and `foundationpose_icp`, settings ZS and FT.

| | |
|---|---|
| Upstream | https://github.com/NVlabs/FoundationPose, pinned to `a1b694b83e633c2cb6115b9063d940a687759392` (main, 2026-04-29) |
| Licence | NVIDIA Source Code License (non-commercial research use only; see `third_party/FoundationPose/LICENSE`). The weights are under the same terms. |
| Weights | Upstream Google Drive folder https://drive.google.com/drive/folders/1DFezOAD0oD1BblsXVxqDsl8fj0qzB82i: refiner `2023-10-28-18-33-37` (68 MB), scorer `2024-01-11-20-02-45` (190 MB). `setup.sh` downloads them with `gdown` (this worked on 2026-09-28). If Drive blocks it, download the two folders by hand into `weights/`. |
| Environment | Docker image `oshd-foundationpose:a1b694b` (`docker/Dockerfile`): CUDA 12.4 devel, Python 3.11, torch 2.5.1+cu124, pytorch3d v0.7.9, nvdiffrast v0.4.0 and upstream `mycpp`, all compiled for sm_89 (RTX 4090). This follows upstream's current conda recipe. The legacy `wenbowen123/foundationpose` image ships Python 3.8, but oshd-bench needs 3.10 or newer. |

## What it does

For every sample of `results/gt/realsense_color/<domain>_<split>.parquet` (the `T` column is never read), `fp_oshd/predict.py` does the following:

1. **Mask.** It takes the highest-scoring detection of the object from `results/detections/realsense_color/<detector>/<det-setting>/<domain>_<split>.parquet` and decodes the COCO RLE `mask_rle`. A sample without a detection gets no prediction, so it counts as a failure.
2. **Input.** It reads the RealSense colour frame and the nearest depth frame, registered into the colour frame by `oshd_bench.sensors.depth_in_camera` (this needs the `realsense_depth` extrinsics). RGB (bilinear), depth and mask (nearest) are undistorted to the pinhole camera with the same K, because FoundationPose assumes a pinhole camera. The output poses are therefore in the real colour optical frame.
3. **FoundationPose.** It calls `FoundationPose.register(K, rgb, depth, mask, iteration=--refine-iters)` **on every frame**. This is model-based pose estimation: global rotation hypotheses, then the refiner, then the scorer. There is **no tracking mode** (`track_one` is never used), so each frame is independent of the previous ones. The refine-iteration count is `--refine-iters` (default 5, as upstream). Frames with fewer than 4 valid depth pixels inside the mask are skipped, because upstream would only guess a translation there.
4. **+ ICP** (row `foundationpose_icp`). This is Open3D point-to-plane ICP, initialised at the FoundationPose pose:
   * Source: 20k points sampled on the CAD surface, keeping those visible at the initial pose (point z-buffer).
   * Target: the depth points inside the detected mask, kept only within 2 cm of the initial model surface.
   * Schedule: correspondence distances 2, 1 and 0.5 cm, each with a Tukey kernel.
   * Rejection: the ICP result is discarded, and the FoundationPose pose kept, when the target inlier ratio at 1 cm is below 0.3 or drops by more than 0.05, or when the pose moves more than half the object diameter or rotates more than 30°. The `.json` records how many frames were accepted (`n_icp_accepted`).
5. **Output.** Poses are written with `oshd_bench.PoseWriter` to `results/predictions/rgbd_realsense/<row>/<setting>/<domain>_<split>.parquet`.

**CAD.** The CAD is `models/CubeSat_U1.obj`, taken through oshd-dataset's `ObjectModel`. It is in metres, `mesh_scale` is applied, and `load_cad` refuses extents that don't look like metres. It is a 27k-vertex, 37k-face shell, and its material colours become vertex colours for FoundationPose's renderer. FoundationPose centres the mesh internally and `register()` returns the pose of the *original* CAD frame, so the output is `T_sensor_model` in the dataset's CAD frame. No symmetry prior is given to FoundationPose, because the marker holder breaks the cube symmetry. The evaluator applies the 24 cube symmetries itself.

**runtime_ms.** It is the detector's `runtime_ms` from the detections file, plus undistortion, plus `register()`, plus ICP for the `_icp` row. It does not include JPEG/PNG decoding or the depth-to-colour registration (the D455 can stream aligned depth on-device).

## ZS / FT

FoundationPose is used as released: the public repo has no training code. As for all training-free, CAD-based methods in the paper, ZS/FT is **the training of the 2D detector** that supplies the mask. That detector is the Mask R-CNN in `baselines/detector`:

* **ZS**: detections from `detections/realsense_color/maskrcnn/ZS/`, the Mask R-CNN trained on the synthetic train split only.
* **FT**: detections from `detections/realsense_color/maskrcnn/FT/`, the same detector fine-tuned on the real train split.

`--detector oracle` uses GT-rendered masks (`oshd-bench oracle-detections`, setting `GT`) for debugging. It writes under the setting you give, for example `ORACLE`, which is not a Table I cell.

## Commands

```bash
cd /home/ricard/oshd_baselines
bash baselines/foundationpose/setup.sh           # clone @a1b694b, weights, docker build (~30 min)

# "train": nothing to train; checks the detector outputs that ZS / FT need
baselines/foundationpose/docker/run.sh python -m fp_oshd.train

# predict (both rows in one pass; docker/run.sh takes work/gpu.lock)
baselines/foundationpose/docker/run.sh python -m fp_oshd.predict --setting ZS --split test --domain real
baselines/foundationpose/docker/run.sh python -m fp_oshd.predict --setting FT --split test --domain real
# S_sim for the ZS rows (needs synthetic test GT + detections):
baselines/foundationpose/docker/run.sh python -m fp_oshd.predict --setting ZS --split test --domain synthetic

uv run --project bench oshd-bench eval
```

Useful flags:

* `--limit N` and `--stride N` subsample the sample list.
* `--refine-iters N` sets the refiner iterations (default 5).
* `--methods foundationpose` writes only the plain row.
* `--detector oracle` uses the GT masks (writes under e.g. `--setting ORACLE`).
* `--det-setting S` reads a detections file of another setting.
* `--min-score` ignores detections below that score.
* `--debug-dir DIR --debug-every N` writes overlays. They show the mask contour in yellow, FoundationPose in green and +ICP in red, each as projected CAD points plus the cube-body box.

### Smoke test on the dev dataset

```bash
cd /home/ricard/oshd_baselines
export OSHD_DATASET_ROOT=$PWD/work/dev_dataset OSHD_BENCH_RESULTS=$PWD/work/dev_results OSHD_BENCHMARK=none
uv run --project bench oshd-bench oracle-detections --sensor realsense_color --split test --domain real
baselines/foundationpose/docker/run.sh python -m fp_oshd.predict --setting ORACLE --detector oracle \
    --limit 20 --stride 25 --debug-dir $PWD/work/debug/foundationpose
# to see the Table I rows fill in, write the same thing as a (fake) ZS file:
baselines/foundationpose/docker/run.sh python -m fp_oshd.predict --setting ZS --detector oracle --limit 20 --stride 25
uv run --project bench oshd-bench eval && grep -i foundationpose $OSHD_BENCH_RESULTS/tables/table1_pose.csv
```

The dev dataset's `realsense_depth` extrinsics are a placeholder, so the depth is not registered correctly there and the numbers mean nothing.

### Smoke-test result (2026-09-28, dev dataset, oracle masks, 20 samples at stride 25)

* 19/20 samples got a pose. One oracle mask had no valid depth.
* Median e_t was 1.4 cm and median e_q 3.8°. 15/19 were within 5 cm.
* The overlays in `work/debug/foundationpose/` line up with the cube.
* **Close range fails.** The failures are frames with the cube closer than about 0.4 m, which is below the D455's minimum depth. There the depth inside the mask is background, so FoundationPose puts the object metres away and the mean e_t blows up (about 4 m).
* **Speed.** `register()` takes about 830 ms per frame (252 hypotheses × 5 refiner iterations), about 1.2 Hz. ICP adds about 10 ms. The full dev test split (526 samples) takes about 8 min.
* **ICP on dev.** On dev, ICP is often rejected or makes things slightly worse, because the depth extrinsics are a placeholder.
* **ICP itself is verified.** On synthetic depth rendered at the GT pose, ICP recovers a 2 cm perturbation to about 1 mm.

## Simulated docking: `sim_zerog` / `docking` / `test`

FoundationPose single-frame RGB-D, CAD-based and training-free, on the eval runs of the Zero-G
lab docking task in simulation: the Floating Platform docks to a satellite mockup or a CubeSat
held by the wall arm, the ceiling arm or floating. This is not a Table I cell; it runs on its own
sample list and writes everything to `results/sim_zerog_docking_test/` in this folder.

| | |
|---|---|
| Split | the frozen **`benchmark_v1`** (oshd-dataset ≥ 1.0.0; Hub revision `8205a11`): `test` = the eval runs, ~20% of the recording runs of each scenario config, whole runs only. 72 trajectories (24 runs x 3 episodes): 51 satellite mockup, 21 CubeSat |
| Samples | every 10th RGB frame (2 Hz of 20 Hz) where the target's semantic mask has ≥ 100 px: 5465 samples |
| Input | sim RGB + depth (320x240, ideal pinhole `K` from `calibration/<traj>.yaml`), no undistortion |
| Mask | the simulator's semantic mask of class `target` (**oracle 2D segmentation**, setting `ORACLE_MASK`). No detector is trained or run |
| CAD | the dataset's target model, `models/sim/<model>/model.obj` on the Hub (`traj.target_mesh()`): `satellite_mockup`, or the run's procedural `cubesat<seed>`, scaled by the run's `scale` |
| GT | `T_cam_target = inv(T_world_cam) @ T_world_target` from the sim mocap (`oshd_dataset.labels.camera_pose`, `traj.body_pose("target")`), in the rgb optical frame and the target's root frame |
| Method | `FoundationPose.register()` on **every frame** from scratch (252 rotation hypotheses, 5 refiner iterations, scorer); no `track_one`, no ICP |

### Step by step

The paths assume this repo at `~/oshd_dataset/oshd_baselines`. Everything else, including the
targets' CAD, comes from the Hub through oshd-dataset.

Needs an NVIDIA GPU with Docker + the NVIDIA container toolkit, and `hf auth login` for the
download.

**1. Install uv** (once):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh     # -> ~/.local/bin/uv
```

**2. Host environment.** This `uv` venv runs the dataset download, the CAD export (`usd-core`)
and the evaluation. It needs `oshd-dataset` ≥ 1.0.0 (sim tasks, `benchmark_v1`), from TestPyPI.
Only `oshd-dataset` itself comes from TestPyPI (`--no-deps`); its dependencies come from PyPI.
`oshd-bench` is installed without dependencies too.

```bash
cd ~/oshd_dataset/oshd_baselines/baselines/foundationpose
uv venv -p 3.11 .venv
uv pip install --python .venv/bin/python --no-deps --index-url https://test.pypi.org/simple/ "oshd-dataset==1.0.0"
uv pip install --python .venv/bin/python "numpy>=1.23" "pyarrow>=14" "huggingface_hub>=0.23" "pillow>=9" \
    "pyyaml>=6" "tqdm>=4.60" trimesh usd-core pandas scipy matplotlib
uv pip install --python .venv/bin/python --no-deps -e ../../bench
.venv/bin/python -c "import oshd_dataset; print(oshd_dataset.__version__)"    # 1.0.0
```

To work on unreleased oshd-dataset changes, install the local checkout instead:
`uv pip install --python .venv/bin/python -e "../../../oshd-dataset[mesh]"`.

**3. The frozen split.** The paper uses `benchmark_v1`: Hub revision `8205a11` and its fixed
train / test lists, shipped inside oshd-dataset. The growing daily split in the Hub's
`splits.json` is not used here.

```bash
.venv/bin/oshd splits --benchmark v1          # sim_zerog/docking: train 300, test 72
```

**4. Download the docking test runs** of the benchmark (about 5 GB, only the modalities used).
`--benchmark v1` pins the download to revision `8205a11`:

```bash
.venv/bin/oshd download --benchmark v1 --domain sim_zerog --task docking --split test \
    -m rgb -m depth -m semantic -m mocap
```

**5. FoundationPose: upstream code, weights, Docker image.** Build the image for your GPU's
compute capability (`nvidia-smi --query-gpu=compute_cap --format=csv`): 8.6 for the RTX 3090
here, 8.9 for a 4090. The build takes about 30 min.

```bash
cd ~/oshd_dataset/oshd_baselines
FP_CUDA_ARCH=8.6 FP_IMAGE=oshd-foundationpose:a1b694b-sm86 bash baselines/foundationpose/setup.sh
```

**6. Target CAD, sample list and ground truth** (host, about 1 min):

```bash
cd baselines/foundationpose
.venv/bin/python -m fp_oshd.sim_prepare --benchmark v1 --domain sim_zerog --task docking --split test --stride 10
#  -> results/sim_zerog_docking_test/{samples,gt}.parquet, objects.json, prepare.json
```

The CAD is the one the dataset publishes for each trajectory. `models/sim/targets.json` maps the
trajectory to its model (`satellite_mockup`, or `cubesat<seed>`: SimForge's procedural CubeSat,
generated from the scenario seed) and scale. The OBJ is in metres, in the frame of the mocap body
`target`, unscaled. `oshd download` fetches the models of the downloaded trajectories; for
`benchmark_v1` they come from Hub revision `22a8431`, because they were uploaded after the freeze.
The ground truth was checked by projecting the scaled CAD with it: 99% of the points fall inside
the target's semantic mask (median, for every target).

**7. Run FoundationPose** in the container (about 1.5 s per frame on the 3090, about 2.3 h for
5465 samples). The image installs oshd-dataset 1.0.0 from TestPyPI. To use a local checkout
instead, set `OSHD_DATASET_SRC=~/oshd_dataset/oshd-dataset`; it is mounted ahead of the image's copy.

```bash
cd ~/oshd_dataset/oshd_baselines
export FP_IMAGE=oshd-foundationpose:a1b694b-sm86
# smoke test (20 samples, overlays):
baselines/foundationpose/docker/run.sh python -m fp_oshd.predict_sim --setting SMOKE --limit 20 --stride 40 --debug-every 5
# full run (--resume continues an interrupted run; it saves every 200 samples):
baselines/foundationpose/docker/run.sh python -m fp_oshd.predict_sim --setting ORACLE_MASK --debug-every 100 --resume \
    > baselines/foundationpose/results/sim_zerog_docking_test/predict_ORACLE_MASK.log 2>&1
```

**8. Evaluate** (host):

```bash
cd baselines/foundationpose
.venv/bin/python -m fp_oshd.eval_sim --run-dir results/sim_zerog_docking_test --setting ORACLE_MASK
```

### Outputs (`results/sim_zerog_docking_test/`)

| file | contents |
|---|---|
| `samples.parquet` / `gt.parquet` | sample list (what inference reads) / plus `T` and `range_m` (only eval reads it) |
| `objects.json`, `prepare.json` | CAD per target (source USD, extent, diameter), sample-list settings |
| `predictions_ORACLE_MASK.parquet` / `.json` | `oshd_bench.PoseWriter` format: `T` (T_cam_target), FoundationPose score, `runtime_ms`; settings + hardware |
| `per_sample_ORACLE_MASK.parquet` | per-sample e_t, e_q, e_q_sym, ADD, ADD_sym, ADD-S, SPEED, range, target, mount |
| `metrics_ORACLE_MASK.json` / `.csv`, `RESULTS_ORACLE_MASK.md` | overall and per target kind, mount and range bin |
| `overlays_ORACLE_MASK/` | every 100th sample: mask contour (yellow), predicted CAD points + box (green) |
| `predict_ORACLE_MASK.log` | the run's log |

**Symmetries.** The satellite mockup (a box with two mirrored solar panels) maps onto itself
under 180° rotations about all three bounding-box axes. Each flip moves the surface by 0.6% of
the diameter (ADD-S), which is sampling noise. The CubeSats are symmetric under the flip about
z only, and only when their parts happen to line up. FoundationPose often returns one of the
flipped poses, which the depth and mask can't tell apart. So `eval_sim` reports e_q and ADD
twice: strict, and `sym`, minimised over the flips measured on each mesh (`--sym-tol`, 1% of the
diameter). ADD-S needs no symmetry list. The texture does break some of the mockup's symmetries;
the strict columns are the upper bound on the error.

### Result (2026-10-20, RTX 3090, oracle masks)

`benchmark_v1` (Hub revision `8205a11`), 72 trajectories, 5465 samples at 2 Hz. Every sample
got a pose. All files are in `results/sim_zerog_docking_test/`; the full table, also per mount, is
`RESULTS_ORACLE_MASK.md`.

| group | samples | e_t med (cm) | e_q med (°) | e_q sym med (°) | 5cm/5° sym | ADD sym < 0.1d | ADD-S < 0.1d | AUC ADD-S | Hz |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **all** | 5465 | 1.19 | 121.9 | 3.03 | 0.590 | 0.652 | 0.830 | 0.670 | 0.64 |
| satellite mockup | 3874 | 1.21 | 167.6 | 3.13 | 0.575 | 0.637 | 0.765 | 0.626 | 0.64 |
| CubeSat | 1591 | 1.14 | 3.03 | 2.83 | 0.627 | 0.689 | 0.989 | 0.778 | 0.83 |
| range < 0.3 m | 563 | 4.84 | 131.2 | 98.6 | 0.002 | 0.067 | 0.647 | 0.296 |  |
| range 0.3–0.5 m | 2364 | 1.75 | 132.9 | 31.8 | 0.376 | 0.480 | 0.739 | 0.540 |  |
| range 0.5–1 m | 1651 | 0.86 | 2.83 | 1.59 | 0.949 | 0.958 | 0.966 | 0.887 |  |
| range 1–2 m | 780 | 0.89 | 5.46 | 1.60 | 0.895 | 0.933 | 0.951 | 0.867 |  |
| range > 2 m | 107 | 1.15 | 106.1 | 2.54 | 0.664 | 0.776 | 0.832 | 0.734 |  |

* **Translation is good everywhere** (median 1.2 cm). **Rotation depends on symmetry and range.**
* **Symmetry.** The strict median rotation error of the mockup is 168°, because FoundationPose
  usually returns one of its 180°-flipped poses, which look the same in shape. Counting the
  measured flips as correct, the median is 3.1°.
* **Range.** From 0.5 to 2 m it works: 90–95% of samples within 5 cm / 5° (sym). The final
  docking approach fails: below 0.5 m the target fills the image (median mask 72–77k of 76.8k
  px). Only one flat face is visible, so a single frame can't fix the rotation. Below 0.3 m only
  0.2% of samples are within 5 cm / 5°. Over a third of the samples (2927 of 5465) are below
  0.5 m, which pulls the overall numbers down. This is the regime where tracking or other
  modalities would have to help.
* **Floating targets** are the hardest mount (sym 5cm/5° 0.53 vs 0.61 for the arms).
* **Speed.** `register()` takes a median of 1.5 s per frame on the RTX 3090 (0.64 Hz); wall time
  was 2.3 h. The mask is an oracle, so these are upper bounds for a detector-based pipeline.

## Files

* `docker/Dockerfile` and `docker/run.sh` are the image and the runner. The runner mounts the repo, the dataset root and its resolved symlinks, `~/.cache/oshd_dataset` and the results dir at their host paths, and mounts `weights/` at `/opt/FoundationPose/weights`. It runs as your uid.
* `fp_oshd/predict.py` is the inference entry point. `fp_oshd/train.py` is the no-op described above.
* `fp_oshd/geometry.py` holds the CAD loading, undistortion, ICP and overlays.
* `fp_oshd/sim_prepare.py`, `fp_oshd/predict_sim.py` and `fp_oshd/eval_sim.py` are the simulated-data pipeline (see above): CAD export and sample list, inference, and metrics.
* `third_party/` and `weights/` are gitignored and filled by `setup.sh`. `cache/` (the exported CAD) and `.venv/` are gitignored too.
