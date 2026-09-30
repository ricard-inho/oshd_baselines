# FoundationPose (RGB-D) and FoundationPose + ICP

Table I, block **RGB-D (RealSense)**: modality `rgbd_realsense`, sensor `realsense_color`, rows
`foundationpose` and `foundationpose_icp`, settings ZS and FT.

| | |
|---|---|
| Upstream | https://github.com/NVlabs/FoundationPose, pinned to `a1b694b83e633c2cb6115b9063d940a687759392` (main, 2026-04-29) |
| Licence | NVIDIA Source Code License (non-commercial research use only; see `third_party/FoundationPose/LICENSE`). The weights are under the same terms. |
| Weights | Upstream Google Drive folder https://drive.google.com/drive/folders/1DFezOAD0oD1BblsXVxqDsl8fj0qzB82i: refiner `2023-10-28-18-33-37` (68 MB), scorer `2024-01-11-20-02-45` (190 MB). `setup.sh` downloads them with `gdown` (this worked on 2026-09-28). If Drive blocks it, download the two folders by hand into `weights/`. |
| Environment | Docker image `pooh-foundationpose:a1b694b` (`docker/Dockerfile`): CUDA 12.4 devel, Python 3.11, torch 2.5.1+cu124, pytorch3d v0.7.9, nvdiffrast v0.4.0 and upstream `mycpp`, all compiled for sm_89 (RTX 4090). This follows upstream's current conda recipe. The legacy `wenbowen123/foundationpose` image ships Python 3.8, but pooh-bench needs 3.10 or newer. |

## What it does

For every sample of `results/gt/realsense_color/<domain>_<split>.parquet` (the `T` column is never read), `fp_pooh/predict.py` does the following:

1. **Mask.** It takes the highest-scoring detection of the object from `results/detections/realsense_color/<detector>/<det-setting>/<domain>_<split>.parquet` and decodes the COCO RLE `mask_rle`. A sample without a detection gets no prediction, so it counts as a failure.
2. **Input.** It reads the RealSense colour frame and the nearest depth frame, registered into the colour frame by `pooh_bench.sensors.depth_in_camera` (this needs the `realsense_depth` extrinsics). RGB (bilinear), depth and mask (nearest) are undistorted to the pinhole camera with the same K, because FoundationPose assumes a pinhole camera. The output poses are therefore in the real colour optical frame.
3. **FoundationPose.** It calls `FoundationPose.register(K, rgb, depth, mask, iteration=--refine-iters)` **on every frame**. This is model-based pose estimation: global rotation hypotheses, then the refiner, then the scorer. There is **no tracking mode** (`track_one` is never used), so each frame is independent of the previous ones. The refine-iteration count is `--refine-iters` (default 5, as upstream). Frames with fewer than 4 valid depth pixels inside the mask are skipped, because upstream would only guess a translation there.
4. **+ ICP** (row `foundationpose_icp`). This is Open3D point-to-plane ICP, initialised at the FoundationPose pose:
   * Source: 20k points sampled on the CAD surface, keeping those visible at the initial pose (point z-buffer).
   * Target: the depth points inside the detected mask, kept only within 2 cm of the initial model surface.
   * Schedule: correspondence distances 2, 1 and 0.5 cm, each with a Tukey kernel.
   * Rejection: the ICP result is discarded, and the FoundationPose pose kept, when the target inlier ratio at 1 cm is below 0.3 or drops by more than 0.05, or when the pose moves more than half the object diameter or rotates more than 30°. The `.json` records how many frames were accepted (`n_icp_accepted`).
5. **Output.** Poses are written with `pooh_bench.PoseWriter` to `results/predictions/rgbd_realsense/<row>/<setting>/<domain>_<split>.parquet`.

**CAD.** The CAD is `models/CubeSat_U1.obj`, taken through pooh-dataset's `ObjectModel`. It is in metres, `mesh_scale` is applied, and `load_cad` refuses extents that don't look like metres. It is a 27k-vertex, 37k-face shell, and its material colours become vertex colours for FoundationPose's renderer. FoundationPose centres the mesh internally and `register()` returns the pose of the *original* CAD frame, so the output is `T_sensor_model` in the dataset's CAD frame. No symmetry prior is given to FoundationPose, because the marker holder breaks the cube symmetry. The evaluator applies the 24 cube symmetries itself.

**runtime_ms.** It is the detector's `runtime_ms` from the detections file, plus undistortion, plus `register()`, plus ICP for the `_icp` row. It does not include JPEG/PNG decoding or the depth-to-colour registration (the D455 can stream aligned depth on-device).

## ZS / FT

FoundationPose is used as released: the public repo has no training code. As for all training-free, CAD-based methods in the paper, ZS/FT is **the training of the 2D detector** that supplies the mask. That detector is the Mask R-CNN in `baselines/detector`:

* **ZS**: detections from `detections/realsense_color/maskrcnn/ZS/`, the Mask R-CNN trained on the synthetic train split only.
* **FT**: detections from `detections/realsense_color/maskrcnn/FT/`, the same detector fine-tuned on the real train split.

`--detector oracle` uses GT-rendered masks (`pooh-bench oracle-detections`, setting `GT`) for debugging. It writes under the setting you give, for example `ORACLE`, which is not a Table I cell.

## Commands

```bash
cd /home/ricard/pooh_baselines
bash baselines/foundationpose/setup.sh           # clone @a1b694b, weights, docker build (~30 min)

# "train": nothing to train; checks the detector outputs that ZS / FT need
baselines/foundationpose/docker/run.sh python -m fp_pooh.train

# predict (both rows in one pass; docker/run.sh takes work/gpu.lock)
baselines/foundationpose/docker/run.sh python -m fp_pooh.predict --setting ZS --split test --domain real
baselines/foundationpose/docker/run.sh python -m fp_pooh.predict --setting FT --split test --domain real
# S_sim for the ZS rows (needs synthetic test GT + detections):
baselines/foundationpose/docker/run.sh python -m fp_pooh.predict --setting ZS --split test --domain synthetic

uv run --project bench pooh-bench eval
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
cd /home/ricard/pooh_baselines
export POOH_DATASET_ROOT=$PWD/work/dev_dataset POOH_BENCH_RESULTS=$PWD/work/dev_results
uv run --project bench pooh-bench oracle-detections --sensor realsense_color --split test --domain real
baselines/foundationpose/docker/run.sh python -m fp_pooh.predict --setting ORACLE --detector oracle \
    --limit 20 --stride 25 --debug-dir $PWD/work/debug/foundationpose
# to see the Table I rows fill in, write the same thing as a (fake) ZS file:
baselines/foundationpose/docker/run.sh python -m fp_pooh.predict --setting ZS --detector oracle --limit 20 --stride 25
uv run --project bench pooh-bench eval && grep -i foundationpose $POOH_BENCH_RESULTS/tables/table1_pose.csv
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

## Files

* `docker/Dockerfile` and `docker/run.sh` are the image and the runner. The runner mounts the repo, the dataset root and its resolved symlinks, `~/.cache/pooh_dataset` and the results dir at their host paths, and mounts `weights/` at `/opt/FoundationPose/weights`. It runs as your uid.
* `fp_pooh/predict.py` is the inference entry point. `fp_pooh/train.py` is the no-op described above.
* `fp_pooh/geometry.py` holds the CAD loading, undistortion, ICP and overlays.
* `third_party/` and `weights/` are gitignored and filled by `setup.sh`.
