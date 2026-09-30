# Target detector: Mask R-CNN (boxes + instance masks)

This 2D detector finds the CubeSat and initialises the model-based pose methods (MegaPose
for RGB and E2VID + MegaPose, FoundationPose for RGB-D). Those methods are training-free, so
**their ZS/FT setting is this detector's training**:

* **ZS**: COCO initialisation, trained on the synthetic train split only (`domain == synthetic`, split `train`).
* **FT**: starts from the ZS weights and is fine-tuned on the real train split (`domain == real`, split `train`).

| | |
|---|---|
| Model | torchvision `maskrcnn_resnet50_fpn_v2` (Li et al., "Benchmarking detection transfer learning with vision transformers", 2021), with new box and mask heads for the `objects` of `configs/bench.yaml` (`cubesat`) |
| Code / licence | torchvision 0.21.0, BSD-3-Clause. Our code is in `src/detector/`. No upstream clone is needed |
| Init weights | `MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1`, https://download.pytorch.org/models/maskrcnn_resnet50_fpn_v2_coco-73cbd019.pth (hash-checked; saved to `weights/torch/hub/checkpoints/`) |
| Trained weights | `weights/<sensor>/{ZS,FT}.pt` plus a `.json` with the settings and the loss log |
| Env | uv project (Python 3.12, torch 2.6.0+cu124) |

## Sensors and resolution

| `--sensor` | input image | network |
|---|---|---|
| `firefly` | 3072x2048 RGB | downscaled 2x to 1536x1024 (min_size 1024) |
| `realsense_color` | 640x480 RGB | torchvision upsamples internally to 800 px short side |
| `events` | E2VID frame, 1280x720 grayscale x3 (`results/e2vid/<domain>_<split>/<traj>/<t_ns>.png`, from `baselines/e2vid`) | native (min_size 720) |

Boxes and masks are always reported in **full-resolution raw (distorted) sensor pixels**.

## Labels

Training uses the GT sample lists `results/gt/<sensor>/<domain>_train.parquet`. Only the
training code reads the `T` column, and only on train splits. For each sample, the CAD
silhouette (`pooh_dataset.labels.render_mask`, which uses the sensor's distortion) is
rendered from the GT pose at the network input resolution. Its tight box becomes the box
label. Samples with fewer than 20 mask pixels are used as negatives. Rendered masks are
cached in `cache/masks/` (set `DETECTOR_CACHE` to move them).

The augmentations are horizontal flip, brightness, contrast, gamma, saturation, random
grayscale and Gaussian noise. The optimiser is SGD with momentum 0.9, AMP, a linear warm-up
and a cosine schedule. The defaults are 12 epochs at lr 0.02 (ZS) and 6 epochs at lr 0.005
(FT), both at batch size 4.

## Output

`predict` writes through `pooh_bench.DetectionWriter` to
`results/detections/<sensor>/maskrcnn/<ZS|FT>/<domain>_<split>.parquet`. The columns are
trajectory, sensor, t_ns, object, box (xyxy), score, mask_rle (COCO uncompressed RLE; decode
with `pooh_bench.predictions.rle_to_mask`) and runtime_ms.

* Each sample of the GT list gets its top-1 detection with score ≥ 0.05 (`--topk`, `--min-score`). Samples without a detection get no row.
* `runtime_ms` covers the detector only: resize, network and mask upsampling. It excludes JPEG/PNG decoding and E2VID. The pose methods add their own time, and for events the E2VID `runtime_ms` from `results/e2vid/<domain>_<split>/index.parquet` as well.
* Inference never reads the `T` column.

## Commands

```bash
cd /home/ricard/pooh_baselines
./baselines/detector/setup.sh
# dev: export POOH_DATASET_ROOT=$PWD/work/dev_dataset POOH_BENCH_RESULTS=$PWD/work/dev_results
# events only: first run baselines/e2vid (reconstruct --all) so the train + test frames exist
cd baselines/detector
L="flock ../../work/gpu.lock"
for S in firefly realsense_color events; do
  $L uv run python -m detector.train   --sensor $S --setting ZS            # synthetic train
  $L uv run python -m detector.train   --sensor $S --setting FT            # ZS -> real train
  for T in ZS FT; do
    $L uv run python -m detector.predict --sensor $S --setting $T --split test --domain real
  done
  $L uv run python -m detector.predict --sensor $S --setting ZS --split train --domain synthetic  # S_sim for the pose rows (use --split test once a synthetic test split exists)
done
```

Smoke-test flags: `train --max-iters N --limit N`, `predict --limit N --debug-dir ../../work/debug`
(which writes overlays of the first `--debug-n` samples).
