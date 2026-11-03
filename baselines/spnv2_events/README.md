# SPNv2 on event time surfaces, simulated docking (`spnv2_events`, Table I row 4)

SPNv2 (Park & D'Amico) trained and tested on **time surfaces of the simulated event camera** of
the `sim_zerog / docking` runs, single target `satellite_mockup`. ROADMAP.md row #4; the RGB
counterpart is row #3 (`baselines/spnv2`, Aman). Setting **`TS`**.

| | |
|---|---|
| Upstream | https://github.com/tpark94/spnv2 pinned at `debee80c43f226a22dce857b0a136de62e9e939d` (2024-09-22), the same commit as `baselines/spnv2`. Paper: Park & D'Amico, *Robust multi-task learning and online refinement for spacecraft pose estimation across domain gap*, Adv. Space Res. 2023 |
| Licence | MIT (`third_party/spnv2/LICENSE.md`, (c) 2022 SLAB Group). The ImageNet EfficientNet-B3 initialisation is torchvision's (BSD-3, ported from timm) |
| Weights | trained here from the ImageNet EfficientNet-B3 backbone (upstream's own initialisation; `setup.sh` fetches it to `weights/torch/`). The trained checkpoint is `work/spnv2_events/runs/TS/model.pth.tar` (not in git, 53 MB state dict + config; ask Matteo) |
| Environment | uv project, Python 3.10, torch 2.5.1+cu124, `albumentations==1.1.0` as upstream. No upstream file is patched: everything that differs lives in `src/spnv2_events/` |
| Data | `benchmark_v3` (oshd 1.1.0, Hub `c8e5ca0`): the 384 `satellite_mockup` runs of `sim_zerog/docking/train` for training, the shared test list `results/sim/sim_zerog_docking_test_v2/` (its 99 mockup runs, 7467 samples) for the numbers |
| Scoring | `oshd-bench sim-eval --subset "object == 'satellite_mockup'" --name TS_mockup` |

## Results (`results/sim_zerog_docking_test_v2/RESULTS_TS_mockup.md`, commit of this PR)

**Paper row (means, rotation up to the measured symmetries): e_t = 6.55 cm, e_q = 15.2°, S = 0.359**,
7467 / 7467 mockup samples predicted (coverage 1), 99 trajectories.

| method, same 7467 samples | input | e_t mean (cm) | e_t med (cm) | e_q sym mean (°) | e_q sym med (°) | S sym | 5cm/5° sym | ADD-S<0.1d | Hz |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SPNv2 `TS` (this folder) | events, 50 ms time surface, oracle mask at train time only | 6.55 | 3.43 | 15.2 | 3.6 | 0.359 | 0.405 | 0.901 | 43 |
| FoundationPose `ORACLE_MASK` (`baselines/foundationpose`) | RGB-D + oracle mask | 3.65 | | 34.8 | | 0.690 | | | |

* **Rotation** is where the time surfaces pay off: the median symmetric rotation error is 3.6°,
  and the mean (15.2°) is driven by the `floating` mount (31.6° mean, 18° median, against ~10° / 3° on the
  arm mounts; not analysed further) and by the two far range bins. Without the
  symmetry reduction the median error is 136°: the network does predict one of the four flip-equivalent
  poses, as the canonicalisation intends, but not always the canonical one.
* **Translation** is worse than FoundationPose's depth-based estimate (6.55 vs 3.65 cm mean) and
  grows with range: 3–4 cm mean below 1 m, 15.7 cm at 1–2 m, 43 cm beyond 2 m (168 samples). The
  monocular depth comes from the EfficientPose regression head (the heatmap PnP translation is not
  used, per upstream's final convention); the median is 3.4 cm.
* The heatmap PnP pose was accepted on 21 % of the samples (`heatmap_pose_rate` in
  `predictions_TS.json`); the other 79 % use the EfficientPose regression for rotation too (fewer than
  5 keypoints above the adaptive threshold, or PnP rejected). Not analysed further.
* **Speed**: 23 ms median per sample on the RTX 4070 (43 Hz in `sim-eval`'s `Hz` column) for the
  time-surface build, warp, network and PnP; reading the 50 ms event window from parquet adds 10 ms
  (median) and is not counted. Predicting the 7467 samples took 4 min.
* Inference reads `samples.parquet` only (`trajectory`, `t_ns`, `scale`, `object`); `gt.parquet`
  is touched by `sim-eval` alone. Twelve overlays (time surface, predicted wireframe, heatmap peaks)
  are in `overlays_TS/`.

Timings (RTX 4070 12 GB, 12-core host): download 12 + 40 GB ≈ 1 h at ~30 MB/s; `sim-prepare` of
the train split ≈ 3 min; time-surface conversion of the 28 814 training samples 85 s with 10
workers (8 GB of PNGs under `work/spnv2_events/data/`); `spnv2ev-check` < 1 min; training 7.0 h;
prediction 4 min; `sim-eval` < 1 min.

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
* **Mask label**: the simulator's semantic mask of class `target` (oracle segmentation, like
  FoundationPose's mask), resampled to 1/4 resolution (160×128). **Box label**: the projected
  scaled CAD vertices, clipped to the image.
* **Training**: upstream's recipe. AdamW, lr 1e-3 stepped ×0.1 at epochs 15 and 18 of 20, AMP,
  batch 4 (upstream: 4 per GPU × 4 GPUs), upstream's augmentations (brightness/contrast, random
  erase, sun flare, blur, noise; no style randomisation). No tuning was done on this input.
  **Run here**: 28 814 time surfaces (the 384 mockup train runs at 2 Hz, every 10th rgb frame, like
  the test list), 7203 it/epoch, 144 060 iterations in **7.0 h on one RTX 4070 12 GB** (175 ms/it,
  5.7 it/s, 8.8 GB of VRAM; batch 8 does not fit). Per-epoch mean losses went hmap 5.60 → 0.78,
  pose 0.93 → 0.23, seg 0.20 → 0.04, cls 0.97 → 0.007, box 0.11 → 0.034 (epochs 1 → 19);
  in epoch 20 one batch produced a non-finite box/pose loss (the AMP scaler skips such steps, the
  heatmap and segmentation means kept improving and every weight of the final checkpoint is finite),
  so the epoch-20 mean of those two terms is `nan` in `train.log`. The checkpoint is the one after
  epoch 20; no model selection was done (no validation split).
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
tests), trailing `KEY VALUE` yacs overrides; `spnv2ev-predict --limit N --pose-source heatmap|regression
--overlays N`. Converted data and checkpoints live under `work/spnv2_events/` (`$SPNV2EV_WORK`).
Delete `work/spnv2_events/data` to regenerate the time surfaces (e.g. after a `--limit` smoke run
fixed `canon.json` from a few samples).

## Files

* `src/spnv2_events/common.py`: paths, the time surface, the virtual camera, the target frame
  (bounding-box corners, flips), canonicalisation, the yacs config.
* `src/spnv2_events/convert.py`: train list → SPEED+-style folder (`spnv2ev-convert`).
* `src/spnv2_events/train.py`, `predict.py`, `check.py`: the driver around upstream's trainer,
  inference on the shared test list, label sanity checks.
* `results/sim_zerog_docking_test_v2/`: `predictions_TS.parquet` + `.json`, `per_sample_*`,
  `metrics_*`, `RESULTS_TS_mockup.md`, `overlays_TS/`.
