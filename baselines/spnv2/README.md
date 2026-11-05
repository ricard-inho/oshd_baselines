# SPNv2 on OSHD (`spnv2`, Table I rows RGB Firefly / RGB RealSense)

The shared-sample-list experiments (Table I row 3, sim docking; Table II, real ↔ lab twin) run
through `spnv2-list`: see [Shared sample lists](#shared-sample-lists-spnv2-list-table-i-row-3-table-ii).
The sections before it describe the earlier real-cube pipeline (`spnv2-train` / `spnv2-predict`).

* **Upstream:** https://github.com/tpark94/spnv2, pinned at commit
  `debee80c43f226a22dce857b0a136de62e9e939d` (2024-09-22). Paper: Park & D'Amico, *Robust
  multi-task learning and online refinement for spacecraft pose estimation across domain gap*,
  Adv. Space Res. 2023.
* **Licence:** MIT (`third_party/spnv2/LICENSE.md`, (c) 2022 SLAB Group). The ImageNet
  EfficientNet-B3 weights are torchvision's (BSD-3, ported from timm).
* **Weights:**
  * The released SPEED+ checkpoints are for the Tango spacecraft (11 keypoints). They can't
    predict our cube, and the OneDrive folder in upstream's README can't be fetched from a
    script: the API answers `unauthenticated` / 403.
  * By default the backbone therefore starts from ImageNet EfficientNet-B3, which is also
    upstream's own initialisation. `setup.sh` downloads it to `weights/torch/`.
  * If you download a SPEED+ checkpoint by hand, pass it with `--init weights/speedplus/<file>`. Every tensor whose name and shape match is loaded: the backbone and most of the heads, but not the 11-channel heatmap output layer.
* **ZS:** trained only on split `train` / domain `synthetic` (the Isaac-Sim renders). It is evaluated on real test data (S_real) and on synthetic test data (S_sim).
* **FT:** starts from the ZS checkpoint and fine-tunes on `train` / `real`. It uses the same virtual camera and symmetry reference as ZS.

## Environment

This is a uv project. It uses Python 3.10–3.11 and torch 2.5.1+cu124. Upstream targets Python 3.7 and torch 1.10, but its code runs unmodified here, with `albumentations==1.1.0` pinned as upstream does. The only side effect is deprecation warnings.

**No upstream file is patched.** The pieces that had to differ live in `src/spnv2_oshd/`:

* A `SPEEDPLUSDataset` subclass derives the image and mask folder names from the input size. Upstream hard-codes `images_768x512_RGB`; see the `TODO` there.
* The train loop is a short single-GPU driver around upstream's `build_spnv2`, `do_train`, `get_optimizer`, `adjust_learning_rate` and `get_scaler`. Upstream's `tools/train.py` is DDP plus validation on the SPEED+ splits.
* Heatmap → pose uses EPnP-RANSAC plus LM refinement, as specified for this benchmark. Upstream uses plain EPnP. Keypoints are taken at the heatmap cell centre, `(x+0.5)·stride`, where upstream uses `x·stride`, which is biased by half a cell.
* ODR reuses upstream's `engine.adapter.do_adapt` and the BN-hook setup from `tools/odr.py`. It passes the hooked BN features with `backbone.`-prefixed names.
  * Upstream hooks `model.backbone.named_modules()` but matches the names against `model.named_modules()`. Those names never agree, so upstream's manual running-statistics update never fires.
  * Only the BN affine update by AdamW would take effect upstream. With our prefix, both updates are applied.
* Upstream's `RandomSunFlare` places its flare circles with `tan(angle)`. Within ~1e-7 rad of vertical a circle centre passes int32 and `cv2.circle` (OpenCV 4.10) raises; this stopped a run after ~1M samples. `spnv2_oshd.lists.clamp_sun_flare` clamps the centres to ±1e6 px at runtime. Such a circle is off the image and was never drawn, so the augmentation is unchanged (checked: identical images on normal draws).

## Data conversion (`spnv2-convert`, run automatically by `spnv2-train`)

The converter turns oshd data into a SPEED+-style folder under `work/spnv2/<dataset>/data/speedplus_<sensor>/`. The CSV format is upstream's `tools/preprocess.py` output.

* **Virtual camera.** Every frame is undistorted and resampled into a fixed pinhole camera per sensor. The result is written as grayscale × 3, following upstream's preprocessing.
  * `firefly`: 3072×2048 → ×0.25 → **768×512**, upstream's input size. The image is area-downscaled first, then undistorted.
  * `realsense_color`: 640×480 → ×1.2 → 768×576, padded to **768×640**. Both sides must be divisible by 128.
  * The virtual K is fixed from the first trajectory's intrinsics and stored in `vcam.json` and in the checkpoint. Other trajectories are warped into this same K, so the regression head's translation stays consistent even when the calibration differs per trajectory.
  * The virtual camera shares the sensor's optical frame, so its poses are `T_sensor_model`.
* **Labels.** Each sample stores:
  * 8 keypoints: the cube-body corners from `objects.yaml`.
  * q and t of `T_cam_net`, where the "net" frame is the CAD frame shifted to the cube centre, c = (0, 0.05, 0). In this frame the symmetries are pure rotations, so t does not depend on which of the 24 equivalent poses is chosen.
  * The bounding box, from the projected mesh.
  * A mask from `oshd.labels.render_mask`, at 1/4 resolution.
* **Symmetry (24-fold cube).** Labels are **canonicalised**:
  * For each ground-truth pose, the converter picks the equivalent `T·S_k` whose rotation, expressed in the viewing-ray (allocentric) frame, is closest to a fixed reference `R_ref`.
  * `R_ref` is the chordal mean of the ZS training orientations on SO(3)/cube (`canon.json`). It is placed at the centre of the data so few samples lie near a cell boundary.
  * Corner identities therefore follow the image appearance. Both the heatmap targets and the regression target use the same canonical pose.
  * Near cell boundaries (±45° in the dominant yaw) identities flip. That is inherent to the approach, and RANSAC absorbs some of it.
  * Evaluation is symmetry-aware (oshd-bench), so whichever equivalent is predicted is fine.
* Delete `work/spnv2/<dataset>/data` to regenerate the data, for example after a `--limit` smoke run fixed `canon.json` from only a few samples.

## Model / training

This is the full SPNv2 configuration from upstream's `experiments/offline_train_full_config_phi3_BN.yaml`. The yaml copy is `configs/spnv2_phi3.yaml`.

* φ = 3, EfficientNet-B3 plus BiFPN, BN backbone and GN heads.
* Three heads: heatmap (8 keypoints), EfficientPose (class, box, 6D rotation, translation, trained with the SPEED loss) and segmentation. Loss factors are 1/1/1.
* Upstream's augmentations: brightness/contrast, random erase, sun flare, blur and noise.
* AdamW with AMP. ZS runs 20 epochs at learning rate 1e-3, stepped down at epochs 15 and 18. FT runs 10 epochs at 1e-4, stepped at epoch 8.
* Batch size is 4 on one GPU (upstream: 4 per GPU × 4 GPUs with SyncBN). Batch 8 at 768×640 needs about 22 GB.
* Style (texture) randomisation is off. It needs a stylised copy of every image, which upstream makes with a separate repository.

## Inference

* **Pose.** Rotation comes from heatmap PnP and translation from the EfficientPose head. This is upstream's final-pose convention, see its `docs/BUGFIX_20230725.md`.
* **Fallback.** The full regression pose is used when the heatmap pose is rejected:
  * fewer than 5 keypoints pass the adaptive threshold (upstream's rule),
  * or RANSAC fails,
  * or the pose falls outside the frustum or beyond 10 m.
* `--pose-source heatmap|regression` uses one head only.
* **Score** is the sigmoid of the best anchor's class score.
* **`runtime_ms`** covers the warp, normalisation, network, post-processing and PnP, with CUDA sync. JPEG decoding is excluded.
* **Overlays** are written to `work/debug/spnv2/`: green is the predicted cube, red dots are the heatmap keypoints (dark red = below threshold), blue is the predicted mask and cyan is the predicted box.
* `--odr` runs Online Domain Refinement on the unlabelled evaluation images first. The upstream defaults are N = 1024 samples, B = 4, LR = 1e-4. Its output goes to setting `<ZS|FT>_ODR`, which is not a table row.
* Inference reads the sample lists without the `T` column.

## Commands

```bash
cd /home/ricard/oshd_baselines
baselines/spnv2/setup.sh
# (dev) export OSH_DATASET_ROOT=$PWD/work/dev_dataset OSHD_BENCH_RESULTS=$PWD/work/dev_results OSHD_BENCHMARK=none
P="uv run --project baselines/spnv2"
L="flock work/gpu.lock"
for S in firefly realsense_color; do
  $L $P spnv2-train --sensor $S --setting ZS          # synthetic train (converts data first)
  $L $P spnv2-train --sensor $S --setting FT          # + real train, from ZS
  for SET in ZS FT; do $L $P spnv2-predict --sensor $S --setting $SET --domain real --split test; done
  $L $P spnv2-predict --sensor $S --setting ZS --domain synthetic --split test   # S_sim
done
uv run --project bench oshd-bench eval
```

Useful flags:

* `spnv2-train`: `--max-iters N` and `--epochs`, `--batch`, `--lr`. `--stride` / `--limit` reduce the training frames. `--init <SPEED+ ckpt>` initialises ZS from a SPEED+ checkpoint. Trailing `KEY VALUE` pairs are yacs overrides.
* `spnv2-predict`: `--limit N` takes an evenly spaced subset. Also `--odr`, `--pose-source` and `--checkpoint`.
* Checkpoints: `work/spnv2/<dataset>/runs/<sensor>/<setting>/model.pth.tar`. Set `$SPNV2_WORK` to use another location.

## Dev smoke test (verified)

Run on the dev dataset (2026-09-28). The numbers are meaningless: placeholder Firefly extrinsics and tiny training runs.

* **ZS.** 300 iterations on "synthetic" trajectory_008 for each sensor: 1354 Firefly and 274 RealSense samples. Firefly used batch 8; RealSense used batch 4 after the OOM.
* **FT.** 100 iterations on real trajectory_007, starting from the ZS checkpoint.
* **Predict.** `--limit 100` on real test trajectory_009 for ZS and FT, for both sensors.
* **ODR.** `--odr --odr-samples 32` on RealSense FT, which writes `FT_ODR`.
* **Eval.** `oshd-bench eval` fills the four `spnv2` rows (coverage 100/1296 and 100/526, about 45 Hz).
  * S_sim is empty because the dev dataset has no synthetic *test* split. `predict` then prints "nothing to predict" and writes no file.
* **Heatmap PnP check.** After this little training the heatmaps are still flat (peaks around 0.05), so every pose came from the regression fallback.
  * PnP on ideal heatmaps generated from the training labels gives 0.1–1.7° and 0.1–0.3 cm. This confirms the keypoint, quaternion and canonical-frame conventions end to end.
* **Labels.** Label overlays in `work/debug/spnv2/label_check_rs.png` show the corners and masks aligned on RealSense. Firefly is misaligned, as expected from its placeholder extrinsics.

## Shared sample lists (`spnv2-list`): Table I row 3, Table II

SPNv2, monocular RGB, trained and tested on the shared oshd-bench sample lists of the repo
(`results/sim/`, `results/real/`). The same code serves three ROADMAP rows:

| row | experiment | train list | test list | network(s) |
|---|---|---|---|---|
| Table I #3 | `sim` | `sim_zerog` docking train, satellite-mockup runs | `results/sim/sim_zerog_docking_test_v2`, `--subset "object == 'satellite_mockup'"` | `satellite_mockup` |
| Table II #6 | `real` | `results/real/real_realsense_color_train_v3` (RealSense colour) | `results/sim/twin_test_v3` | `antena`, `antena_1_solar` |
| Table II #7 | `twin` | `sim-prepare --benchmark v3 --domain twin --task all --split train` | `results/real/real_realsense_color_test_v3` | `antena`, `antena_1_solar` |

Code: `src/spnv2_oshd/lists.py` (`spnv2-list convert | train | predict | labels | trajectories`),
configuration: `configs/experiments.yaml`. Training, the network, the loss and inference are the
ones described above (upstream's full φ = 3 configuration and its 20-epoch schedule; its
augmentations except style (texture) randomisation, see "Model / training"; heatmap PnP rotation + regression translation, regression fallback).

### Data

All numbers use the frozen **`benchmark_v3`** (oshd 1.1.0 from PyPI, Hub `r3m3c3/oshd` revision
`c8e5ca0`). Table I uses v3's copy of the `benchmark_v2` docking split: the committed
`results/sim/sim_zerog_docking_test_v2` list, as every Table I row. Checked for the files Table I
reads (483 trajectories): rgb, semantic, mocap, metadata and the mockup CAD are byte-identical
at the v2 (`df3530b`) and v3 (`c8e5ca0`) revisions; the calibration differs only by the new
`lidar.points_origin` key; oshd 1.0.0 → 1.1.0 only adds the twin domain, the v3 split and the sim
lidar origin fix; and the training list rebuilt with `--benchmark v3` has the same 57 789 mockup
rows (samples and labels) as the `--benchmark v2` one used here.

### Design decisions

* **One network per target.** A keypoint network regresses the keypoints of one object, so the
  sim docking split (22 targets) is restricted to the satellite mockup (7467 of 12 237 test
  samples), as CONTRIBUTING.md proposes. For Table II each real mockup gets a network; the real
  mockup and its lab twin share one CAD frame (`antena` / `real_antena`), so the same network
  serves both names. A test sample goes to the network whose `objects` hold its `object`.
* **32 keypoints**, farthest-point samples of the CAD surface, in the frame centred at the CAD
  bounding-box centre (where `sim-eval` measures e_t). SPEED+ uses 11 for Tango, but docking is
  close: with 11, only 55% of the mockup test samples have the 5 keypoints in view that the
  heatmap pose needs (24: 99.1%, 32: 99.6%).
* **Symmetry.** The satellite mockup maps onto itself under 180° flips about its three box axes
  (measured by `sim-eval`; `convert` checks the yaml against the CAD). Training labels are
  canonicalised over that group, as for the cube above (closest to the chordal-mean reference in
  the viewing-ray frame), so the keypoint identities follow the image. The real mockups have no
  measured symmetry: no canonicalisation.
* **Scale.** The sim targets are scaled per run (`scale` in `samples.parquet`, which CONTRIBUTING
  lists as an input; FoundationPose scales its CAD by it too). One image can't tell size from
  distance, so the network is trained on the pose of the unit-scale CAD that gives the same image
  (`t / scale`) and the predicted translation is multiplied by the sample's `scale`.
* **Virtual camera** `sim_rgb_x2`: the sim / twin `rgb` camera (320x240, f = 227.3 px) ×2 =
  640x480, padded to **640x512** (both sides divisible by 128). The RealSense colour frames
  (1280x720 in benchmark_v3, not 640x480 as CONTRIBUTING.md says; FoV 89.7° x 58.6°) are
  undistorted and area-resampled into the same camera, i.e. cropped to the twin's field of view,
  so the real ↔ twin networks see the same geometry in both domains (98% of the real training
  samples keep the mockup centre in view). Grayscale x 3, as upstream.
* **Masks** for the segmentation head: the simulator's semantic class `target` (sim, twin) or the
  CAD rendered at the ground-truth pose (real). On sim training frames the two agree to IoU 0.985
  (mean, 151 samples). Samples whose projected box is under 2 px or behind the camera are
  dropped from training (4 of 57 789 for `sim`).
* **No test labels at inference.** `predict` reads `samples.parquet` (without `T` / `range_m`) and
  the frames; `gt.parquet` of a test list is read only by `oshd-bench sim-eval`.

### Validation before the long runs

* **Label overlays** (`spnv2-list labels --exp <exp>` → `work/debug/spnv2/labels_<exp>_<target>.jpg`):
  mask, box, CAD box at the label pose and the stored and re-projected keypoints agree for all
  five networks.
* **Ideal-heatmap oracle**: Gaussian heatmaps made from the converted training keypoints, through
  the inference code (heatmap PnP, frame and scale conversion), written with `PoseWriter` and
  scored with `oshd-bench sim-eval` against the training list's labels (~400 samples per network;
  coverage < 1 = heatmap pose rejected: fewer than 5 keypoints in view, RANSAC failure or
  outside the frustum, which the regression fallback covers at test time):

  | network | samples | coverage | e_t mean (cm) | e_q mean (°) |
  |---|---:|---:|---:|---:|
  | sim / satellite_mockup | 402 | 0.983 | 0.36 | 0.2 |
  | real / antena | 416 | 0.983 | 0.29 | 0.4 |
  | real / antena_1_solar | 409 | 0.971 | 0.38 | 0.5 |
  | twin / antena | 406 | 0.966 | 0.15 | 0.3 |
  | twin / antena_1_solar | 406 | 0.919 | 1.09 | 0.4 |

* **Smoke runs** of convert → train → predict → `sim-eval` on small subsets, and a sanity
  evaluation of the first sim epoch (298 spread test samples: e_t 4.90 cm, e_q 11.0°, S 0.250).

### Step by step

From the repo root, on one GPU (here an RTX 4070, 12 GB, with an i5-14600K and 32 GB RAM). Run
GPU jobs under `flock work/gpu.lock`. `export VIRTUAL_ENV=` if another venv is active.

```bash
cd bench && uv sync && cd ..                     # oshd 1.1.0 (PyPI) + oshd-bench
baselines/spnv2/setup.sh                         # upstream at debee80, uv env, EfficientNet-B3 weights
P="uv run --project baselines/spnv2 spnv2-list"
L="flock work/gpu.lock"

# 1. Data (benchmark_v3). Table I: only the satellite-mockup docking runs (train 384, test 99
#    trajectories; 27 + 6.7 GB, ~5 + 1.5 min here). Table II: the twins and the real recordings.
for S in test train; do
  uv run --project bench oshd download --benchmark v3 \
      $($P trajectories --exp sim --split $S --task docking | sed 's/^/-t /') -m rgb -m semantic -m mocap
done
uv run --project bench oshd download --benchmark v3 --domain twin -m rgb -m semantic -m mocap  # 8 GB, ~2.5 min
uv run --project bench oshd download --benchmark v3 --domain real -m realsense_color -m mocap  # 11 GB, ~2.5 min

# 2. Training lists (the test lists and the real train list are committed)
uv run --project bench oshd-bench sim-prepare --benchmark v2 --domain sim_zerog --task docking \
    --split train --stride 5          # 45 s: 57 789 samples (4 Hz), 384 runs -> results/sim/sim_zerog_docking_train_v2
uv run --project bench oshd-bench sim-prepare --benchmark v3 --domain twin --task all \
    --split train --stride 5          # 24 332 samples, 160 runs -> results/sim/twin_train_v3

# 3. Check the labels (optional), convert + train (train converts first)
$P labels --exp sim
$L $P train --exp sim                  # convert 1:53 min, train 14.2 h (20 epochs x 14 446 it, 5.66 it/s)

# 4. Predict the test list (samples.parquet only) and score it
$L $P predict --exp sim --run-dir results/sim/sim_zerog_docking_test_v2     # 2.8 min, 7467 samples
uv run --project bench oshd-bench sim-eval --run-dir results/sim/sim_zerog_docking_test_v2 \
    --pred baselines/spnv2/results/sim_zerog_docking_test_v2/predictions_SIM.parquet \
    --subset "object == 'satellite_mockup'" --name SIM_mockup               # 15 s
```

* `train --resume` continues after the last saved epoch (checkpoints are written every epoch, with
  the optimizer and AMP-scaler state). `--limit N`, `--max-iters N`, `--epochs` are for smoke
  tests. `predict --adapt adabn|odr` is the unsupervised-adaptation variant of Table II (see
  there); without it `predict` is unchanged.
* Converted data and checkpoints: `work/spnv2/<dataset>/{data,runs/<exp>/<target>}/` (`$SPNV2_WORK`
  overrides). `predict` writes `results/<test list>/predictions_<EXP>.parquet` + `.json` here and
  12 overlays (`--overlays`): CAD box at the predicted pose (green), heatmap keypoints (red), mask
  (blue), detected box (cyan).

**This run.** The first `sim` training stopped in epoch 19 of 20 on the `RandomSunFlare`
overflow (see Environment). After the fix, epochs 19 and 20 (lr 1e-5) were resumed from the
epoch-18 weights with `train --resume`. That checkpoint predates the optimizer-state saving, so
AdamW's moment estimates restarted for those two epochs; the schedule, data and iteration count
(288 920) are upstream's. The epoch-18 weights scored e_t 2.16 cm, e_q 5.8°, S 0.125 (same
protocol), so the last two epochs change little.

### Result: sim docking, satellite mockup (Table I, row 3; 2026-11-04, RTX 4070)

`results/sim_zerog_docking_test_v2/`: 99 test trajectories, 7467 mockup samples at 2 Hz. Every
sample got a pose (heatmap PnP on 6862, regression fallback on 605). Full table:
`RESULTS_SIM_mockup.md`.

**Paper row** (means over the samples, rotation up to the measured symmetries):
**e_t = 2.13 cm, e_q = 5.8°, S = 0.125.** Medians: 0.49 cm, 0.94°.

Against FoundationPose on the same samples (`baselines/foundationpose`, RGB-D with oracle masks,
`RESULTS_ORACLE_MASK_mockup.md`):

| group | samples | SPNv2 e_t med / mean (cm) | FP e_t med / mean (cm) | SPNv2 e_q sym med / mean (°) | FP e_q sym med / mean (°) | SPNv2 S sym | FP S sym | SPNv2 5cm/5° sym | FP 5cm/5° sym |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **all** | 7467 | 0.49 / 2.13 | 1.22 / 3.65 | 0.94 / 5.8 | 3.38 / 34.9 | 0.125 | 0.690 | 0.833 | 0.560 |
| arm (ceiling) | 2684 | 0.47 / 1.60 | 1.11 / 2.87 | 0.80 / 2.0 | 3.03 / 36.3 | 0.054 | 0.703 | 0.940 | 0.574 |
| arm (wall) | 2939 | 0.40 / 1.28 | 1.09 / 2.73 | 0.75 / 1.5 | 3.20 / 36.1 | 0.039 | 0.698 | 0.960 | 0.562 |
| floating | 1844 | 1.18 / 4.27 | 1.78 / 6.24 | 5.28 / 18.3 | 4.00 / 30.8 | 0.365 | 0.657 | 0.473 | 0.535 |
| range < 0.3 m | 796 | 0.48 / 0.91 | 3.58 / 6.11 | 1.62 / 6.3 | 88.7 / 79.8 | 0.148 | 1.615 | 0.838 | 0.010 |
| range 0.3–0.5 m | 3100 | 0.41 / 0.77 | 1.84 / 3.81 | 1.31 / 7.3 | 78.0 / 55.7 | 0.150 | 1.085 | 0.808 | 0.292 |
| range 0.5–1 m | 2294 | 0.44 / 0.74 | 0.84 / 1.13 | 0.52 / 3.9 | 1.68 / 6.2 | 0.078 | 0.124 | 0.890 | 0.934 |
| range 1–2 m | 1109 | 1.17 / 4.98 | 0.88 / 4.41 | 0.80 / 3.9 | 1.55 / 6.3 | 0.099 | 0.136 | 0.844 | 0.907 |
| range > 2 m | 168 | 6.37 / 33.24 | 1.15 / 18.42 | 1.99 / 12.9 | 2.25 / 17.2 | 0.367 | 0.380 | 0.399 | 0.702 |

* **Close range is where SPNv2 wins.** Below 0.5 m (52% of the samples) FoundationPose's single
  frame can't fix the rotation of a target that fills the image; SPNv2, trained on this target
  at these ranges, keeps a median of 1.3–1.6° and 81–84% within 5 cm / 5°.
* **Far targets are its weak spot.** At > 1 m the mockup covers few pixels of the 640x512 input,
  and the depth error grows with range: the 128 samples with e_t > 10 cm (1.7%) have a median mask
  of 2551 px (all samples: 53 274 px), 116 of them at ≥ 1 m, and they carry 57% of the summed
  e_t (mean without them: 0.92 cm). FoundationPose, with depth, is better there.
* **Floating targets** are the hardest mount (e_q sym mean 18.3°, 47% within 5 cm / 5°).
* **Symmetry.** The strict rotation error has a median of 168°: the network predicts one of the
  mockup's flipped poses, chosen by appearance through the canonicalised labels, which is the
  pose the strict metric doesn't credit. The paper row uses the symmetric error, as for
  FoundationPose.
* **Speed.** 50 Hz per frame (warp, network, PnP; `runtime_ms`), 2.8 min for the 7467 samples
  including frame decoding. FoundationPose: 0.67 Hz on an RTX 3090.
* SPNv2 here is RGB only and needs no mask at test time; FoundationPose uses depth and the
  simulator's oracle mask.

Outputs (`results/sim_zerog_docking_test_v2/`):

| file | contents |
|---|---|
| `predictions_SIM.parquet` / `.json` | `oshd_bench.PoseWriter` format: `T` (T_cam_target), class score, `runtime_ms`; checkpoint, upstream commit, virtual camera, heatmap-pose rate |
| `per_sample_SIM_mockup.parquet` | per-sample errors from `sim-eval` |
| `metrics_SIM_mockup.json` / `.csv`, `RESULTS_SIM_mockup.md` | overall, per mount and per range bin |
| `overlays_SIM/` | 12 spread test samples |

### Result: real ↔ lab twin (Table II)

Commands (after steps 1–2 above):

```bash
# row 6: train on real (RealSense colour, 64 runs), test on the twin and, as the in-domain reference, on real
$L $P train --exp real                 # convert 0:54 + 1:38 min, train 2.1 h (antena) + 3.9 h (antena_1_solar)
$L $P predict --exp real --run-dir results/sim/twin_test_v3                   # 73 s, 3038 samples
$L $P predict --exp real --run-dir results/real/real_realsense_color_test_v3  # 6 min, 14 594 samples
uv run --project bench oshd-bench sim-eval --run-dir results/sim/twin_test_v3 \
    --pred baselines/spnv2/results/twin_test_v3/predictions_REAL.parquet
uv run --project bench oshd-bench sim-eval --run-dir results/real/real_realsense_color_test_v3 \
    --pred baselines/spnv2/results/real_realsense_color_test_v3/predictions_REAL.parquet

# row 7: train on the twin (sim rgb, 160 runs), test on real and, as the in-domain reference, on the twin
$L $P train --exp twin                 # convert 0:18 + 0:16 min, train 3.0 h + 3.0 h
$L $P predict --exp twin --run-dir results/real/real_realsense_color_test_v3  # 6 min, 14 594 samples
$L $P predict --exp twin --run-dir results/sim/twin_test_v3                   # 69 s, 3038 samples
uv run --project bench oshd-bench sim-eval --run-dir results/real/real_realsense_color_test_v3 \
    --pred baselines/spnv2/results/real_realsense_color_test_v3/predictions_TWIN.parquet
uv run --project bench oshd-bench sim-eval --run-dir results/sim/twin_test_v3 \
    --pred baselines/spnv2/results/twin_test_v3/predictions_TWIN.parquet
```

| row | train → test | e_t (cm) | e_q (°) | S | in-domain reference (same network, own test split) |
|---|---|---:|---:|---:|---|
| 6 | real → twin | 48.75 | 110.3 | 2.428 | real test: 21.71 cm, 46.2°, S 0.944 |
| 7 | twin → real | 124.28 | 110.1 | 2.673 | twin test: 4.75 cm, 10.1°, S 0.236 |

Both directions fail: the networks don't transfer between the real mockup and its lab twin,
with RGB alone and upstream's augmentations (style randomisation off, see "Model / training").

#### Row 6: trained on real, tested on the lab twin (2026-11-04, RTX 4070)

`results/twin_test_v3/RESULTS_REAL.md`: the 40 twin test trajectories, 3038 samples (every 10th
frame). Every sample got a pose (heatmap PnP on 28%, regression fallback on the rest).

**Paper row: e_t = 48.75 cm, e_q = 110.3°, S = 2.428** (the real mockups have no measured
symmetry, so strict = sym).

| test set | samples | e_t med / mean (cm) | e_q med / mean (°) | S | 5cm/5° | heatmap PnP |
|---|---:|---:|---:|---:|---:|---:|
| **twin test** (row 6) | 3038 | 41.94 / 48.75 | 112.3 / 110.3 | 2.428 | 0.000 | 28% |
| twin, `real_antena` | 1511 | 25.89 / 39.21 | 131.2 / 116.1 | 2.48 | | |
| twin, `real_antena_1_solar` | 1527 | 54.20 / 58.19 | 108.6 / 104.6 | 2.38 | | |
| real test (in-domain reference) | 14 594 | 16.19 / 21.71 | 30.5 / 46.2 | 0.944 | 0.106 | 77% |
| real, `antena` | 3741 | 32.29 / 36.46 | 47.5 / 56.1 | 1.19 | 0.223 | |
| real, `antena_1_solar` | 10 853 | 14.46 / 16.63 | 26.7 / 42.7 | 0.86 | 0.066 | |
| real *train* samples (fit check, every 100th) | 494 | 1.02 / 1.84 | 0.8 / 1.7 | 0.044 | 0.943 | 98% |

* **No transfer.** On the twin, a median rotation error of 112° is chance level (a uniformly
  random rotation gives a median of 132° and a mean of 126°). The overlays show why: on the rendered frames the segmentation
  head fires on the background, the keypoint heatmaps scatter, and the heatmap pose is rejected
  on 72% of the samples.
* **The twin test views are mostly outside the real training distribution.** The twin is seen from
  0.26–1.66 m (5–95%, median 0.84 m; 63% under 1 m), the real training runs from 0.79–2.07 m
  (median 1.54 m; 7% under 1 m). The rendered mockup is also bright and textureless next to the
  dark, specular real one.
* **Even within the real domain the network barely generalises.** It fits its own training
  samples to 1.84 cm / 1.7° (the same `predict` + `sim-eval` path, run on every 100th sample of
  `results/real/real_realsense_color_train_v3`), so the pipeline is right. But on the 16 held-out
  real runs it reaches 21.7 cm / 46°. The errors depend strongly on the run: medians go from
  2.5 cm / 1.5° (`trajectory_023`) and 3.9 cm / 1.5° (`trajectory_062`) up to 66 cm / 82°
  (`trajectory_025`). The real training split has 23 (`antena`) and 41 (`antena_1_solar`) runs
  of 30 Hz video, i.e. few distinct views. A test frame's rotation is a median 19.8° (`antena`)
  and 8.0° (`antena_1_solar`) away from the nearest training rotation, and the overlays show
  correct masks and boxes but wrong poses.

Outputs: `results/twin_test_v3/` (row 6) and `results/real_realsense_color_test_v3/` (in-domain
reference), with `predictions_REAL.parquet` / `.json`, `RESULTS_REAL.md`, `metrics_REAL.json` /
`.csv`, `per_sample_REAL.parquet` and 12 overlays each.

#### Row 7: trained on the twin, tested on real (2026-11-04, RTX 4070)

`results/real_realsense_color_test_v3/RESULTS_TWIN.md`: the 16 real test trajectories, 14 594
samples (every frame). Every sample got a pose (heatmap PnP on 37%, regression fallback on the
rest).

**Paper row: e_t = 124.28 cm, e_q = 110.1°, S = 2.673.**

| test set | samples | e_t med / mean (cm) | e_q med / mean (°) | S | 5cm/5° | heatmap PnP |
|---|---:|---:|---:|---:|---:|---:|
| **real test** (row 7) | 14 594 | 137.27 / 124.28 | 115.6 / 110.1 | 2.673 | 0.000 | 37% |
| real, `antena` | 3741 | 107.98 / 113.23 | 152.6 / 137.7 | 3.085 | 0.000 | |
| real, `antena_1_solar` | 10 853 | 141.43 / 128.08 | 96.3 / 100.6 | 2.531 | 0.000 | |
| twin test (in-domain reference) | 3038 | 1.84 / 4.75 | 1.9 / 10.1 | 0.236 | 0.646 | 87% |
| twin, range 0.5–2 m | 2100 | 1.34–1.67 / 2.87–5.81 | 0.9–1.3 / 4.1–6.9 | 0.11–0.15 | 0.76–0.82 | |
| twin, range < 0.5 m | 880 | 2.04–2.27 / 4.11–4.56 | 9.1–10.5 / 17.0–23.1 | 0.40–0.52 | 0.28–0.37 | |

* **The twin network is good in its own domain.** 160 training runs of a randomised simulator
  give it the view coverage the real split lacks: a median of 1.8 cm / 1.9° on the 40 held-out
  twin runs (82% within 5 cm / 5° between 1 and 2 m). Close range (< 0.5 m, where the mockup
  overfills the 70° field of view) is its hardest part.
* **It does not transfer to real.** Every one of the 16 real runs fails (median e_t 81–156 cm,
  median e_q 39–170° per run). The failure is systematic: on the dark real frames the
  segmentation head marks most of the image as the target and the detected box covers nearly
  the whole frame, so the network reads the mockup as huge and close. The predicted range has a
  median of 0.34 m (5–95%: 0.13–0.73 m) for a true median of 1.51 m.
* **This is the reverse of row 6.** Each network only works on the image statistics it was
  trained on: rendered, bright, textureless mockup (twin) vs dark, specular mockup under a sun
  lamp (real). The twin training ranges (0.23–1.69 m, 5–95%) cover the real test ranges (median
  1.51 m), so here distance isn't the cause.

Outputs: `results/real_realsense_color_test_v3/` (row 7) and `results/twin_test_v3/` (in-domain
reference), with `predictions_TWIN.parquet` / `.json`, `RESULTS_TWIN.md`, `metrics_TWIN.json` /
`.csv`, `per_sample_TWIN.parquet` and 12 overlays each.

#### Unsupervised adaptation without retraining (variant, not the paper row)

The same networks, adapted before predicting on **1024 unlabelled images of the target domain's
train split** (`adapt_list` in the yaml: `twin_train_v3` for `real`, `real_realsense_color_train_v3`
for `twin`; never the test list, which `predict` refuses), evenly spaced over each mockup's samples.
Test lists, `PoseWriter` and `sim-eval` are unchanged; outputs are `*_ADABN` / `*_ODR` next to the
plain ones.

* `--adapt adabn`: AdaBN (Li et al., 2016). The backbone's 145 BatchNorm layers re-estimate
  their running statistics (cumulative average, batches of 8); no weight changes. The heads use
  GroupNorm and are untouched. About 20 s per network.
* `--adapt odr`: upstream's online domain refinement with its settings
  (`odr_phi3_B4_N1024.yaml`): entropy minimisation of the segmentation head, updating only the
  backbone BatchNorm layers, 1024 images, gradient batches of 4, AdamW lr 1e-4. About 40 s per network.

```bash
for A in adabn odr; do
  $L $P predict --exp real --run-dir results/sim/twin_test_v3 --adapt $A                  # 1.5 min
  $L $P predict --exp twin --run-dir results/real/real_realsense_color_test_v3 --adapt $A  # 7 min
  uv run --project bench oshd-bench sim-eval --run-dir results/sim/twin_test_v3 \
      --pred baselines/spnv2/results/twin_test_v3/predictions_REAL_${A^^}.parquet
  uv run --project bench oshd-bench sim-eval --run-dir results/real/real_realsense_color_test_v3 \
      --pred baselines/spnv2/results/real_realsense_color_test_v3/predictions_TWIN_${A^^}.parquet
done
```

| row | variant | e_t med / mean (cm) | e_q med / mean (°) | S | heatmap PnP |
|---|---|---:|---:|---:|---:|
| 6 real → twin | plain (paper row) | 41.94 / 48.75 | 112.3 / 110.3 | 2.428 | 28% |
| | + AdaBN | 36.92 / 44.83 | 110.7 / 112.1 | 2.532 | 37% |
| | + ODR | 35.77 / 44.10 | 106.7 / 110.7 | 2.488 | 36% |
| 7 twin → real | plain (paper row) | 137.27 / 124.28 | 115.6 / 110.1 | 2.673 | 37% |
| | + AdaBN | 123.85 / 116.70 | 107.8 / 104.3 | 2.529 | 44% |
| | + ODR | 129.43 / 119.24 | 100.5 / 100.7 | 2.479 | 42% |

* **Checks.** AdaBN on the twin network's own domain (twin train images → twin test) leaves it
  where it was (4.70 cm / 10.1° / S 0.234 against 4.75 / 10.1 / 0.236), so the re-estimation
  itself doesn't damage a network. ODR on the twin network for real images halves the
  segmentation entropy on 64 held-out real train images (0.155 → 0.067) and moves the BatchNorm
  running means by 5%.
* **Small gains, no transfer.** Both methods lower the mean e_t by 4–10%; the mean e_q drops by
  5–9% on row 7 and not at all on row 6. Rotation stays near chance and no sample is within
  5 cm / 5°. On real frames the twin network still marks
  most of the frame as target (56% of the pixels after ODR, 73% before) and puts the mockup at a
  median 0.38 m (true 1.51 m; plain 0.34 m); the keypoints stay scattered in the overlays.
  Re-normalising the features doesn't help when the features themselves don't carry over between
  the rendered and the real mockup: closing this gap needs training-time changes (style
  randomisation, background and photometric augmentation), not test-time ones.

### Limitations of the Table II rows (6 and 7)

What the numbers show: with RGB alone and this training recipe, SPNv2 doesn't transfer between
the real mockups and their lab twins, in either direction. What they don't show is how far a
stronger recipe or method would get. Read them with these limits:

**Training recipe**

* **Style (texture) randomisation is off.** Upstream's full configuration turns it on for
  synthetic → lab transfer, and it targets the appearance gap measured here. It needs a stylised
  copy of every training image (a separate upstream repository), which wasn't set up. So rows 6
  and 7 are a pessimistic SPNv2: upstream's architecture, losses and schedule with its other
  augmentations, not its domain-gap recipe.
* **No validation split and no tuning.** Every network is the last (20th) epoch of upstream's
  schedule. No hyperparameter or checkpoint was chosen on any result, so nothing is tuned to the
  test lists, but nothing is tuned for this data either.
* **One run per network** (seed 2021 as upstream, but cuDNN non-deterministic, so a rerun won't
  match bit for bit; no repeats): no variance estimate. A retraining costs
  about 6 h per direction on the RTX 4070.
* **Grayscale input** (as upstream) in both domains: colour cues are discarded.
* **Single frame.** The test lists are videos, but every frame is predicted on its own; there's no
  tracking or temporal smoothing.

**Data**

* **The real training split has few distinct views.** It has 64 runs (23 `antena`, 41
  `antena_1_solar`) of 30 Hz video: 49 382 frames, used at 15 Hz. The real network fits its
  training frames (1.84 cm / 1.7°) but reaches 21.7 cm / 46° on the 16 held-out real runs, and a
  test frame's rotation is a median 8–20° from the nearest training rotation. Row 6 therefore
  mixes the domain gap with too little real data, and its in-domain reference should be reported
  next to it.
* **The twin is seen closer than the real mockup.** Twin test: 0.26–1.66 m (5–95%), median
  0.84 m. Real training: 0.79–2.07 m, median 1.54 m. 63% of the twin test frames are under 1 m,
  against 7% of the real training frames, so row 6 is also a range shift. Row 7 isn't: the twin
  training ranges (0.23–1.69 m) cover the real test (median 1.51 m).
* **The RealSense frames are cropped to the twin's field of view.** Real frames are 1280x720 at
  89.7° x 58.6°; both domains go into one 640x512 virtual camera at the twin's 70.3° x 55.7°. The
  mockup centre stays in view in 97.9% of the real train and test frames. The other 2.1% are still
  predicted and scored. They are a disadvantage for the real side that the twin side doesn't have.
* **The two domains get their training masks differently.** Twin masks are the simulator's
  `target` class; real masks are the CAD rendered at the ground-truth pose. On sim frames the two
  agree to IoU 0.985, so the difference is small but not zero.
* **Real ground truth** is mocap-based. It was checked visually here (training-label overlays line
  up with the images), not independently. The real network's training fit (1.8 cm / 1.7°)
  includes any label noise. CONTRIBUTING.md reports the RealSense ground truth as aligned.
* **Unequal test sets.** The twin test has every 10th frame of 40 runs (3038 samples, half per
  mockup). The real test has every frame of 16 runs (14 594 samples, 74% `antena_1_solar`).
  Consecutive frames are strongly correlated, so the real test mean is dominated by a few long
  `antena_1_solar` runs, and the per-mockup rows are worth reading.

**Protocol**

* **The network is told which mockup is in view.** A test sample goes to that mockup's network
  by the `object` column of `samples.parquet`. This is benchmark metadata, used as FoundationPose
  uses the object's CAD, but it is not a detection.
* **The adaptation variants** (`*_ADABN`, `*_ODR`) are reported next to the paper rows, never
  instead of them. They use 1024 unlabelled images of the target domain's train split per
  network, never test frames, with upstream's ODR settings unchanged. They only adapt the
  backbone's normalisation layers, which is why they can't fix a gap in the learned features.
* **Events are not covered.** Rows 6 and 7 for events are Lara's. If the paper sets RGB and
  events side by side, the plain rows are the comparable ones; style randomisation, if added
  later, has no direct event equivalent.
* **Hardware.** Timings are for an RTX 4070 (12 GB); the ROADMAP's estimates are for an RTX 3090.

**What would most likely change the numbers** (each a separate, labelled variant on the same test
lists and `sim-eval`): upstream's style randomisation, label-exact augmentation (in-plane rotation
about the optical axis, background replacement through the masks, stronger photometric changes),
and, for row 6, more distinct real views.
