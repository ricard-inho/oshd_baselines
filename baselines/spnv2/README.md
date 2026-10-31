# SPNv2 on OSHD (`spnv2`, Table I rows RGB Firefly / RGB RealSense)

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
