# SPNv2 on event time surfaces, simulated docking (`spnv2_events`, Table I row 4)

SPNv2 (Park & D'Amico) trained and tested on **time surfaces of the simulated event camera** of
the `sim_zerog / docking` runs, single target `satellite_mockup`. ROADMAP.md row #4; the RGB
counterpart is row #3 (`baselines/spnv2`, Aman). Setting **`TS`**.

| | |
|---|---|
| Upstream | https://github.com/tpark94/spnv2 pinned at `debee80c43f226a22dce857b0a136de62e9e939d` (2024-09-22), the same commit as `baselines/spnv2`. Paper: Park & D'Amico, *Robust multi-task learning and online refinement for spacecraft pose estimation across domain gap*, Adv. Space Res. 2023 |
| Licence | MIT (`third_party/spnv2/LICENSE.md`, (c) 2022 SLAB Group). The ImageNet EfficientNet-B3 initialisation is torchvision's (BSD-3, ported from timm) |
| Weights | trained here from the ImageNet EfficientNet-B3 backbone (upstream's own initialisation; `setup.sh` fetches it to `weights/torch/`). The trained checkpoint is `work/spnv2_events/runs/TS/model_best.pth.tar` (not in git, 53 MB state dict + config; ask Matteo). The first run (hard canonical labels, upstream losses, last epoch, no validation) is kept as `work/spnv2_events/runs/TS_canon/` |
| Environment | uv project, Python 3.10, torch 2.5.1+cu124, `albumentations==1.1.0` as upstream. No upstream file is patched: everything that differs lives in `src/spnv2_events/` |
| Data | `benchmark_v3` (oshd 1.1.0, Hub `c8e5ca0`): the 384 `satellite_mockup` runs of `sim_zerog/docking/train`, 352 for training and 32 held out for validation (`configs/val_trajectories.txt`), the shared test list `results/sim/sim_zerog_docking_test_v2/` (its 99 mockup runs, 7467 samples) for the numbers |
| Scoring | `oshd-bench sim-eval --subset "object == 'satellite_mockup'" --name TS_mockup` |

## Results (`results/sim_zerog_docking_test_v2/`, commit of this PR)

**Paper row (`RESULTS_TS_mockup.md`; means, rotation up to the measured symmetries):
e_t = 6.73 cm, e_q = 17.0°, S = 0.387**, 7467 / 7467 mockup samples predicted (coverage 1),
99 trajectories, 43 Hz. The `TS` model is the one selected on the held-out validation trajectories
(epoch 19 of 20); nothing was chosen on the test split.

| run, same 7467 samples | labels / losses | checkpoint | e_t mean (cm) | e_t med | e_q sym mean (°) | e_q sym med | S sym | 5cm/5° sym | ADD-S<0.1d | heatmap PnP accepted |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **`TS`** (paper row) | canonical labels + symmetry-aware losses | best val epoch (19) | 6.73 | 3.49 | 17.0 | 4.2 | 0.387 | 0.365 | 0.897 | 22 % |
| `TS_canon` (first run) | canonical labels, upstream losses | last epoch (20), no validation | 6.55 | 3.43 | 15.2 | 3.6 | 0.359 | 0.405 | 0.901 | 21 % |
| FoundationPose `ORACLE_MASK` (`baselines/foundationpose`, RGB-D + oracle mask) | | | 3.65 | | 34.8 | | 0.690 | | | |

Pose-source ablation (`spnv2ev-predict --pose-source`; `RESULTS_TS[_canon]_{heatmap,regression}_mockup.md`).
Error means are over the predicted samples; the heatmap-only rows cover only the samples where PnP
was accepted (their `coverage`):

| run | pose source | coverage | e_t mean (cm) | e_q sym mean (°) | e_q sym med | S sym |
|---|---|---:|---:|---:|---:|---:|
| `TS` | fused (upstream's final convention) | 1.000 | 6.73 | 17.0 | 4.2 | 0.387 |
| `TS` | regression head only | 1.000 | 6.73 | 15.3 | 4.0 | 0.358 |
| `TS` | heatmap PnP only | 0.221 | 78.0 | 16.4 | 3.4 | 2.213 |
| `TS_canon` | fused | 1.000 | 6.55 | 15.2 | 3.6 | 0.359 |
| `TS_canon` | regression head only | 1.000 | 6.55 | 14.5 | 3.5 | 0.346 |
| `TS_canon` | heatmap PnP only | 0.208 | 37.7 | 12.4 | 3.0 | 0.940 |

What the numbers say:

* **Rotation** is where the time surfaces pay off against FoundationPose (17.0° vs 34.8° mean,
  4° median). The mean is driven by the `floating` mount (33.9° mean, 20.9° median in `TS`, against
  ~11° / 3.4° on the two arm mounts) and by the far ranges. Without the symmetry reduction the
  median rotation error is ~177° (`TS`) / 136° (`TS_canon`): the network predicts one of the four
  flip-equivalent poses, as intended, and with the symmetry-aware losses it settled on one flip for
  almost every sample.
* **Translation** is worse than FoundationPose's depth-based estimate (6.7 vs 3.65 cm mean; 3.5 cm
  median) and grows with range: 3–4 cm mean below 1 m, 16.6 cm at 1–2 m, 48 cm beyond 2 m (168
  samples). It comes from the EfficientPose regression head in every setting (upstream's convention;
  the PnP translation of 8 corners is far worse, see the heatmap-only rows).
* **Symmetry-aware losses and validation did not change the test numbers** beyond what looks like
  run-to-run variation (`TS` vs `TS_canon`: +0.2 cm, +1.8°, +0.03 S, with the symmetry-aware run on
  352 instead of 384 training runs). The hard canonicalisation of the labels was therefore not the
  bottleneck. The `TS` row is kept as the paper row because its checkpoint was selected on held-out
  data; `TS_canon` is the ablation.
* **Heatmap PnP is accepted on ~21 % of the samples** and this is set by the range, not the labels:
  per range bin the acceptance is 3 % below 0.3 m, 6 % at 0.3–0.5 m, 22 % at 0.5–1 m, 75 % at 1–2 m,
  63 % beyond 2 m (`TS`; `TS_canon` within 6 points). Below 1 m the bounding-box corners are outside
  the 640×512 frame or the peaks are too weak for the adaptive threshold (upstream's 5-keypoint
  rule). Where PnP is accepted, its rotation is better than the regression head's at 1–2 m (7.0°
  vs 13.1°) and worse below 1 m, so upstream's fusion (PnP rotation whenever accepted) is slightly
  worse overall than the regression head alone (17.0° vs 15.3°). We report upstream's convention;
  a range-gated fusion would be test-set tuning.
* **Validation curve (`work/spnv2_events/runs/TS/val.json`, 800 samples of the 32 held-out
  runs)**: S_sym 0.744 (epoch 1) → 0.469 (5) → 0.416 (12) → 0.363 (16, after the first lr drop)
  → 0.346 (19, best) → 0.349 (20); e_t 11.6 → 6.3 cm, e_q_sym 33.6° → 14.9°. Epochs 16–20 are
  within 0.02 S of each other, so the model-selection step is not what moved the numbers either.
* **Speed**: 23 ms median per sample on the RTX 4070 (43 Hz in `sim-eval`'s `Hz` column) for the
  time-surface build, warp, network and PnP; reading the 50 ms event window from parquet adds
  10 ms (median) and is not counted. Predicting the 7467 samples takes 4 min.
* Inference reads `samples.parquet` only (`trajectory`, `t_ns`, `scale`, `object`); `gt.parquet`
  is touched by `sim-eval` alone. Twelve overlays of `TS` (time surface, predicted wireframe,
  heatmap peaks) are in `overlays_TS/`.

Timings (RTX 4070 12 GB, 12-core host): download 12 + 40 GB ≈ 1 h at ~30 MB/s; `sim-prepare` of
the train split ≈ 3 min; time-surface conversion of the 26 408 training samples 81 s with 10
workers (8 GB of PNGs under `work/spnv2_events/data/`); `spnv2ev-check` < 1 min; training 6.9 h
(incl. 28 s of validation per epoch); prediction 4 min per setting; `sim-eval` < 1 min.

## Method

SPNv2's full configuration (φ = 3, EfficientNet-B3 + BiFPN, heatmap + EfficientPose + segmentation
heads, `configs/spnv2_phi3.yaml` = upstream's `offline_train_full_config_phi3_BN.yaml` on one GPU)
with these inputs and labels:

* **Input: a time surface** of the events of `[t − 50 ms, t)`, `t` the sample's `t_ns` (the
  benchmark's event window). Three channels, built by `common.time_surface_image`:
  * channel 0: negative events, `exp((t_last − t) / τ)` with τ = 25 ms (window / 2),
  * channel 1: the same for positive events,
  * channel 2: per-pixel event count of both polarities, `log1p(n) / log1p(16)`, clipped to 1.

  Pixels without an event are 0. The 320×240 image is resampled ×2 into a fixed **640×512** virtual
  pinhole camera (640×480 plus 16 black rows top and bottom; sides must be multiples of 128 for the
  P7 stride), stored as PNG and normalised with ImageNet statistics like upstream's 3-channel
  grayscale frames. The sim event camera is pixel-aligned with the sim `rgb` camera (same K and
  pose), so the virtual camera is the `rgb` optical frame and the predicted pose is directly the
  sample's `T_cam_target`.
* **Target frame and keypoints.** The 8 keypoints are the **corners of the CAD's bounding box**
  (`models/sim/satellite_mockup/model.obj`, extent 0.31 × 1.30 × 0.30 m, unscaled). Each run renders
  the CAD at a `scale` in 0.8–1.2 (a column of the sample list, not a label), so the 3D keypoints
  of a sample are `scale × corners`. The regressed translation is the (scaled) bounding-box
  centre, the point whose distance `sim-eval` reports as `e_t`.
* **Symmetry.** `sim-eval` measures the mockup's symmetries as the three 180° flips about its
  bounding-box axes (`Rx180, Ry180, Rz180`, see any `metrics_*.json`) and reports `e_q` / `S` up
  to them. Labels are **canonicalised** to the one of the 4 equivalent poses whose rotation,
  expressed in the viewing-ray (allocentric) frame, is closest to a fixed reference `R_ref`, the
  chordal mean of the training orientations on SO(3)/{flips} (`canon.json`). Corner identities
  therefore follow the image appearance; the identities flip near the cell boundaries, which
  RANSAC absorbs in part. Same construction as `baselines/spnv2` for the 24-fold cube.
* **Symmetry-aware losses** (`symloss.py`, default; `--no-sym-loss` for upstream's): the labels
  stay canonical, but the heatmap MSE and the SPEED pose loss are, per sample, the min over the 4
  flip-equivalent targets (heatmap channels permuted as the flipped corners, rotation target
  `F_k @ quat2dcm(q)` since upstream's target is `R_cam_net^T`; translation is flip invariant). So
  a label near a canonicalisation boundary does not penalise the equivalent the network picks.
  `scripts/test_symloss.py` checks the conventions on real labels (CPU).
* **Mask label**: the simulator's semantic mask of class `target` (oracle segmentation, like
  FoundationPose's mask), resampled to 1/4 resolution (160×128). **Box label**: the projected
  scaled CAD vertices, clipped to the image.
* **Training**: upstream's recipe. AdamW, lr 1e-3 stepped ×0.1 at epochs 15 and 18 of 20, AMP,
  batch 4 (upstream: 4 per GPU × 4 GPUs), upstream's augmentations (brightness/contrast, random
  erase, sun flare, blur, noise; no style randomisation). No tuning was done on this input.
  **First run (`TS_canon`, hard canonical labels, upstream losses)**: 28 814 time surfaces (the 384 mockup train runs at 2 Hz, every 10th rgb frame, like
  the test list), 7203 it/epoch, 144 060 iterations in **7.0 h on one RTX 4070 12 GB** (175 ms/it,
  5.7 it/s, 8.8 GB of VRAM; batch 8 does not fit). Per-epoch mean losses went hmap 5.60 → 0.78,
  pose 0.93 → 0.23, seg 0.20 → 0.04, cls 0.97 → 0.007, box 0.11 → 0.034 (epochs 1 → 19);
  in epoch 20 one batch produced a non-finite box/pose loss (the AMP scaler skips such steps, the
  heatmap and segmentation means kept improving and every weight of the final checkpoint is finite),
  so the epoch-20 mean of those two terms is `nan` in `train.log`. The checkpoint is the one after
  epoch 20; no model selection was done (no validation split).
  **Paper-row run (`TS`, symmetry-aware losses, validation)**: 26 408 time surfaces of the 352
  training runs, 6602 it/epoch, 132 040 iterations in **6.9 h** (176 ms/it, same speed: the extra
  loss terms are negligible); all per-epoch losses finite (hmap 0.85, pose 0.26, seg 0.04 at
  epoch 20). Best validation epoch 19 (`model_best.pth.tar`).
* **Validation / model selection**: 32 of the 384 train trajectories (`scripts/make_val_split.py`,
  seed 0) are left out of the converted training data. After every epoch the checkpoint is run
  through `spnv2ev-predict` + `sim-eval` on up to 800 evenly spaced samples of them
  (`<work>/val/sim_zerog_docking_train_v3_val/`), logged to `train.log` and `<out>/val.json`; the
  epoch with the lowest `speed_sym_mean` is copied to `model_best.pth.tar`, which `spnv2ev-predict`
  uses by default (`checkpoint` / `checkpoint_epoch` / `val` in the predictions metadata).
* **Inference** (`spnv2ev-predict`): upstream's final-pose convention (`docs/BUGFIX_20230725.md`):
  rotation from the heatmap keypoints by EPnP-RANSAC + LM refinement (keypoints at the heatmap
  cell centre), translation from the EfficientPose head; the full regression pose when fewer than
  5 keypoints pass the adaptive threshold, PnP fails, or the pose is outside the frustum or beyond
  10 m. Every mockup sample gets a prediction (coverage 1). `runtime_ms` covers the time-surface
  build, the warp, normalisation, the network, post-processing and PnP, with CUDA sync; reading
  the 50 ms event window from parquet is excluded (its median is in the predictions `.json`).

## Commands (download to `sim-eval`)

```bash
cd oshd_baselines
cd bench && uv sync && cd ..                                   # oshd 1.1.0 + oshd-bench
bash baselines/spnv2_events/setup.sh                            # upstream clone, uv env, ImageNet weights (~3 min)

# data: the docking test runs (events only, 12 GB) and the 384 mockup train runs (rgb, semantic, mocap, events; 40 GB)
uv run --project bench oshd download --benchmark v3 --domain sim_zerog --task docking --split test -m events -m mocap
uv run --project bench python baselines/spnv2_events/scripts/mockup_train_runs.py > /tmp/mockup_train.txt   # the 384 run names
uv run --project bench oshd download --benchmark v3 --domain sim_zerog $(sed 's/^/-t /' /tmp/mockup_train.txt) \
    -m rgb -m semantic -m mocap -m events

# training sample list (gt.parquet of the train split; only the downloaded runs are listed)
uv run --project bench oshd-bench sim-prepare --domain sim_zerog --task docking --split train
#  -> results/sim/sim_zerog_docking_train_v3/ (not committed)

P="uv run --project baselines/spnv2_events"
L="flock work/gpu.lock"
$P spnv2ev-check --run-dir results/sim/sim_zerog_docking_train_v3 --n 8    # label overlays + PnP round trip (work/debug/spnv2_events)
$L $P spnv2ev-train --run-dir results/sim/sim_zerog_docking_train_v3     # converts the data first
$L $P spnv2ev-predict --run-dir results/sim/sim_zerog_docking_test_v2                     # -> results/sim_zerog_docking_test_v2/predictions_TS.parquet
uv run --project bench oshd-bench sim-eval --run-dir results/sim/sim_zerog_docking_test_v2 \
    --pred baselines/spnv2_events/results/sim_zerog_docking_test_v2/predictions_TS.parquet \
    --subset "object == 'satellite_mockup'" --name TS_mockup
```

Useful flags: `spnv2ev-train --max-iters N --limit N --stride N --epochs N --batch N --lr X` (smoke
tests), `--no-sym-loss` (upstream's losses), `--no-val` / `--val-limit N` (per-epoch validation, default
800 samples), trailing `KEY VALUE` yacs overrides; `spnv2ev-predict --checkpoint P --limit N
--pose-source heatmap|regression --overlays N` (default checkpoint `model_best.pth.tar`, else
`model.pth.tar`). Smoke test with two validation rounds (`--max-iters` caps the total, so `--limit`
makes epochs short; it rewrites `labels.csv`, the next full run regenerates it):
`$L $P spnv2ev-train --limit 120 --epochs 2 --max-iters 60 --val-limit 40 --out work/spnv2_events/runs/SMOKE_SYM`. Converted data and checkpoints live under `work/spnv2_events/` (`$SPNV2EV_WORK`).
Delete `work/spnv2_events/data` to regenerate the time surfaces (e.g. after a `--limit` smoke run
fixed `canon.json` from a few samples).

## Files

* `src/spnv2_events/common.py`: paths, the time surface, the virtual camera, the target frame
  (bounding-box corners, flips), canonicalisation, the yacs config.
* `src/spnv2_events/convert.py`: train list minus the validation trajectories → SPEED+-style folder
  (`spnv2ev-convert`).
* `src/spnv2_events/symloss.py`: the symmetry-aware heatmap and pose losses.
* `src/spnv2_events/validate.py`: the validation run-dir and the per-epoch scoring.
* `src/spnv2_events/train.py`, `predict.py`, `check.py`: the driver around upstream's trainer,
  inference on the shared test list, label sanity checks.
* `configs/val_trajectories.txt` (from `scripts/make_val_split.py`): the 32 held-out trajectories.
* `scripts/test_symloss.py`: convention checks of the symmetric losses on real labels.
* `results/sim_zerog_docking_test_v2/`: `predictions_TS.parquet` + `.json`, `per_sample_*`,
  `metrics_*`, `RESULTS_TS_mockup.md`, `overlays_TS/`.
