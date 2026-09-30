# Voxel-grid keypoint network (ours): Table I, Events rows

Method key `event_keypoints`, modality `events`, sensor `events`, settings ZS and FT.

* **Upstream:** none. There is no public event-based spacecraft pose network with weights, so this network is ours. It is built from standard parts: torchvision ResNet-18 (Xiao et al. "SimpleBaseline" deconv heads), the CenterNet focal loss, and OpenCV EPnP + RANSAC + LM.
* **Licence:** this code follows the repo licence. The ImageNet ResNet-18 initialisation comes from torchvision (BSD-3-Clause), https://download.pytorch.org/models/resnet18-f37072fd.pth, and `setup.sh` puts it in `weights/`.
* **Trained checkpoints:** `weights/trained/{ZS,FT}.pt`.

## Method

1. **Input.** A voxel grid (5 bins, bilinear in time, polarity ±1, identical to `pooh_dataset.events.voxel_grid`) of the events in `[t − 50 ms, t)`. It is built on the GPU and standardised over its non-zero voxels.
2. **Stage A (target localisation).** A ResNet-18 with 3 deconvolutions runs on the 1/2-resolution grid (640×360, padded to 384). It outputs a centre heatmap (focal loss) and the log side of the square box around the projected cube corners, at stride 8 in full-resolution pixels.
3. **Stage B (keypoints).** A 256×256 `roi_align` crop of the full-resolution grid, with side 1.3 × the predicted box, goes through a second ResNet-18 + 3 deconvolutions. It outputs 64×64 heatmaps for 9 keypoints: the 8 cube-body corners (`objects.yaml` keypoints) and the symmetry centre. It is trained with an MSE loss against Gaussians with σ = 1.5. Peaks are refined by a 5×5 soft-argmax.
4. **Pose.** `cv2.solvePnPRansac` (EPnP, reprojection threshold 4 % of the box side, minimum 2 px) with the events intrinsics and distortion, then `solvePnPRefineLM` on the inliers. The score is the detection score × the mean inlier heatmap peak. Samples with no detection (peak < 0.1) or no valid PnP get no row, so they count against coverage.
5. **Symmetry.** The cube is 24-fold symmetric (`pooh_bench.symmetry`). Before rendering the keypoint labels, the GT pose is canonicalised to the symmetric equivalent `T·S_k` whose rotation in the event-camera frame is closest to the fixed reference `R_ref = I` (`common.canonical_pose`). A corner's label therefore depends only on the cube's appearance from the camera, i.e. which camera-frame octant it points to. The predicted pose is one symmetric equivalent, and the evaluator minimises over the group. Near the boundaries between canonical cells (rotations about 45° away from `R_ref`), labels can swap between nearby frames. The centre keypoint and RANSAC make PnP tolerate the occasional swapped corner.
6. **Training.** Both stages are trained jointly on the same batches.
   * Stage B sees GT crops with scale jitter ×[0.85, 1.25] and shift ±12 % of the side, so it tolerates stage-A errors.
   * Augmentation: random event drop (keep 50–100 %) and uniform noise events (0–2 % of the pixels).
   * Optimiser: AdamW with a OneCycle cosine schedule, bf16 autocast, batch 16.
   * **ZS:** synthetic train only (`results/gt/events/synthetic_train.parquet`), from ImageNet initialisation. Defaults are 20k iterations at lr 5e-4, about 0.28 s/it at batch 16 on the shared 4090, so ~1.5 h.
   * **FT:** initialised from the ZS checkpoint, then fine-tuned on real train. Defaults are 6k iterations at lr 1e-4.
   * Training reads the `T` column of train splits only. `predict` reads the sample list without `T`.
7. **runtime_ms.** Covers GPU voxelisation (both resolutions), stage A, the crop, stage B and PnP, for one sample at a time on an RTX 4090. The events are already in memory; the median parquet read time is recorded separately in the prediction `.json`.

## Commands

```bash
cd baselines/event_keypoints && ./setup.sh
cd ../.. && export POOH_DATASET_ROOT=... POOH_BENCH_RESULTS=...   # omit both for the real benchmark
cd baselines/event_keypoints
flock ../../work/gpu.lock uv run ekp-train --setting ZS            # synthetic train
flock ../../work/gpu.lock uv run ekp-train --setting FT            # + real train, from ZS.pt
for S in ZS FT; do
  flock ../../work/gpu.lock uv run ekp-predict --setting $S --split test --domain real
done
flock ../../work/gpu.lock uv run ekp-predict --setting ZS --split test --domain synthetic   # S_sim / gap
flock ../../work/gpu.lock uv run ekp-debug --setting FT --n 4      # overlays -> work/debug/events/event_keypoints
cd ../.. && uv run --project bench pooh-bench eval
```

`--limit N` runs N evenly spaced samples. `--iters` and `--lr` override the training defaults.
