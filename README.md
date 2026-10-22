# pooh_baselines

Baselines for the POOH-D dataset paper (Tables I–III, Fig. 1). This is the mother repo:

* `bench/` is the shared `pooh-bench` package. It holds the evaluation sample lists with ground truth, the file formats, the metrics and the table writer.
* `baselines/<name>/` holds one baseline each. Every baseline has its **own** environment (a uv project or a Docker image), because the upstream codes need incompatible CUDA/PyTorch stacks.

```
bench/                 pooh-bench: gt.py, metrics.py, predictions.py, flow_gt.py, evaluate.py
configs/bench.yaml     which predictions fill which table cell, metric settings, Fig. 1 bins
baselines/
  detector/            Mask R-CNN target detector (sim -> ZS, + real -> FT), feeds the model-based methods
  megapose/            MegaPose (RGB, CAD-based, training-free)  -> RGB rows, E2VID + MegaPose
  foundationpose/      FoundationPose RGB-D (+ ICP), Docker image -> RGB-D rows
  spnv2/               SPNv2 trained on renders of the cube      -> RGB rows
  lidar_registration/  FPFH + TEASER++ + ICP (training-free)     -> LiDAR row
  geotransformer/      GeoTransformer (uv; CPU-only C++ extension) -> LiDAR rows
  e2vid/               E2VID event -> frame reconstruction       -> E2VID + MegaPose
  event_keypoints/     voxel-grid keypoint network (ours)        -> Events rows
  eraft/               E-RAFT event optical flow                 -> Table III
  policy_replay/       frozen PPO on estimator-substituted observations -> Table II
scripts/               dev dataset, run-everything script
results/               gt/, predictions/, detections/, replay/, flow/, tables/
```

## Pipeline

All paper numbers use the frozen benchmark **`benchmark_v1`** of pooh-dataset (≥ 0.5.0, from
TestPyPI): the data of Hub revision `8205a11` (2195 sim + 3 real trajectories) and its fixed train
/ test split. `configs/bench.yaml` (`dataset.benchmark: v1`) makes every `ds.split()` of the
bench and the baselines use it. Download its data with `--benchmark v1`, which pins the
revision. Daily uploads after the freeze don't change it.

```bash
cd bench && uv sync && cd ..                  # pooh-dataset 0.5.0 from TestPyPI
uv run --project bench pooh download --benchmark v1 --domain all   # or narrow: -m ..., --split, --task
uv run --project bench pooh splits --benchmark v1
uv run --project bench pooh-bench gt          # results/gt/<sensor>/<domain>_<split>.parquet
# per baseline: see baselines/<name>/README.md  (setup.sh, then train / predict)
uv run --project bench pooh-bench eval        # results/tables/*.csv|json|tex|dat
```

`pooh-bench eval` writes one file per table or figure panel:

| file | contents |
|---|---|
| `table1_pose.csv` / `.json` / `.tex` | e_t (cm), e_q (°) (plus medians in the CSV), S_sim, S_real, sim-to-real gap, ADD(-S) AUC, rate (Hz), coverage, keyed by modality, method and setting |
| `fig1a_range.csv` / `.dat` | S_real per range bin, one column per modality group, plus `<group>_glare` |
| `fig1b_angrate.csv` / `.dat` | S_real per bin of relative angular rate (log-spaced bins) |
| `table2_replay.csv` / `.tex` | mean and P95 ‖Δa‖ per modality |
| `table3_flow.csv` / `.tex` | EPE, 1PE, 3PE, AE per setting |

The `.tex` files are table bodies in the template's row order, with `\tbd` wherever a value is missing. The `.dat` files paste into `\pgfplotstableread{...}`. The Fig. 1 bin edges live in `configs/bench.yaml`.

## Definitions

* **Ground truth.** The pose is `T_sensor_model` = inv(T_world_ref · T_ref_sensor) · T_world_cubesat · T_body_model. It comes from mocap, the rig calibration and `models/objects.yaml`, through `pooh_dataset.labels`.
* **Samples.** For cameras, every frame where the cube projects into the image. For the lidar, every scan where the cube is inside the Mid-360 field of view and range. For events, reference times at 20 Hz, each using the events in `[t − 50 ms, t)`.
* **Errors.** Rotation errors are minimised over the cube's 24 rotational symmetries. Translation errors are measured at the cube centre, which every symmetric pose maps to the same point (`bench/src/pooh_bench/symmetry.py`).
* **SPEED score.** `e_t/‖t_gt‖ + e_q[rad]`, with the SPEED+ precision thresholds.
* **ADD(-S) AUC.** Uses thresholds from 0 to 10 cm. Frames without a prediction count as failures.
* **Coverage.** The fraction of samples with a prediction. e_t, e_q and S are averaged over the predicted samples, so always read them together with coverage.
* **Rate.** 1000 / median(`runtime_ms`) on the machine recorded in the prediction's `.json` (an RTX 4090 here). It covers the whole pipeline: detection, reconstruction, pose.
* **ZS vs FT.**
  * ZS means trained only on the synthetic split (domain `synthetic`), or training-free, then evaluated on real data.
  * FT means additionally fine-tuned on the real train split.
  * For the training-free, CAD-based methods (MegaPose, FoundationPose), the only learned part specific to our data is the 2D detector that initialises them. For these, ZS/FT is the detector's training.
  * The pose networks themselves are not re-trained: FoundationPose's public repo has no training code.

## Changes from the first experiment draft (and why)

* **RGB rows and E2VID use MegaPose instead of FoundationPose.** FoundationPose needs a depth image at inference. It has no RGB-only mode.
* **SPNv2 "ZS" is trained on our synthetic renders of the cube.** The SPEED+ weights are for the Tango spacecraft, so they can't estimate our target.
* **The E-RAFT FT training loop is written here.** The upstream repo only has inference code.
* **The voxel-grid keypoint network is ours.** There is no public event-based spacecraft pose network with weights.

## Baseline contract

Every `baselines/<name>/` provides:

1. A `README.md` covering: the upstream repo and pinned commit, the licence, where the weights come from, and the exact commands.
2. An environment: either `pyproject.toml` (a uv project that depends on `pooh-bench` through a path source) or a `Dockerfile` with `docker/run.sh`. The Docker option mounts the repo, the dataset and `results/`.
3. A `setup.sh` that clones upstream code into `third_party/` at a pinned commit and downloads weights into `weights/`. Both folders are gitignored.
4. Entry points to `train` (for ZS and FT, where applicable) and to `predict --setting ZS|FT --split test --domain real|synthetic [--limit N]`. `predict` writes through `pooh_bench.PoseWriter` to `prediction_path(results_dir(), <modality>, <method>, <setting>, <domain>, <split>)`.

The GT sample lists in `results/gt/` define exactly which samples to run on. Inference code must **never** read the `T` column; only training code on the train split may.

## Development before the calibration is complete

```bash
uv run --project bench python scripts/make_dev_dataset.py
export POOH_DATASET_ROOT=$PWD/work/dev_dataset POOH_BENCH_RESULTS=$PWD/work/dev_results POOH_BENCHMARK=none
uv run --project bench pooh-bench gt
```

This builds a symlinked copy of the real dataset with **placeholder** extrinsics for Firefly, depth, lidar and events, and placeholder event-camera intrinsics. It also marks trajectory_008 as "synthetic". Use it to exercise every code path. The numbers it produces are meaningless.

GPU jobs on the shared machine run under `flock work/gpu.lock <cmd>`, so trainings don't compete for the 24 GB.
