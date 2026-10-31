# E-RAFT: event optical flow on the target (Table III)

* **Upstream:** https://github.com/uzh-rpg/E-RAFT, pinned at `c58ce0524ea0ebfa9849991caafb547f44fe9bfd` (2022-10-08), cloned by `setup.sh` into `third_party/E-RAFT`. Only `model/` and `utils/image_utils.py` are imported. The network code is used unmodified.
* **Licence:** MIT (upstream `LICENSE`).
* **Weights:** the official DSEC checkpoint, https://download.ifi.uzh.ch/rpg/ERAFT/checkpoints/dsec.tar (64 MB, checked reachable 2026-09-28), goes to `weights/dsec.tar`. The FT checkpoint is written to `weights/trained/FT.tar`.
* **Environment:** uv project (torch 2.5.1 cu124). Pure PyTorch, so no CUDA extension is needed.

## Settings

| setting | weights |
|---|---|
| ZS | DSEC checkpoint as-is (E-RAFT "standard", not warm-start) |
| FT | DSEC checkpoint fine-tuned on the **real train** split with target-masked GT flow (`eraft-train`) |

There is no synthetic stage. The DSEC checkpoint is the zero-shot model.

## Samples, inputs, ground truth

* **Flow pairs.** For every reference time `t` of `results/gt/events/<domain>_<split>.parquet` (20 Hz), the flow goes from `t0 = t − 100 ms` to `t1 = t`. Only `trajectory`, `t_ns` and `object` are read; the `T` column is never used. Pairs without mocap at `t0` or `t1` are skipped.
* **DSEC input conventions** (reproduced from `loader_dsec.py` / `VoxelGrid`):
  * `image1` is a 15-bin voxel grid of the events in `[t0 − 100 ms, t0)`, and `image2` covers `[t0, t1)`.
  * Time is normalised per window by the first and last event.
  * Polarity is ±1, splatted trilinearly in x, y and t.
  * Each grid is normalised by the mean and std of its non-zero voxels.
  * 12 iterations, and the last prediction is used.
* **Resolution.** DSEC is 640×480; our Prophesee is 1280×720. By default (`--scale 0.5`), event coordinates are mapped to 640×360 with `u' = (u + 0.5)/2 − 0.5`. These float coordinates are splatted bilinearly, which is how E-RAFT treats DSEC's rectified float coordinates. This keeps DSEC's 640 px width and roughly its pixel-flow range. The network pads to 640×384 internally. The predicted flow is upsampled bilinearly to 1280×720 and multiplied by 2. `--scale 1.0` runs at full resolution instead.
* **No rectification.** The events are not rectified, because the GT flow is defined in the raw, distorted pixel grid of the `events` intrinsics.
* **GT.** `oshd_bench.flow_gt.target_flow_at(traj, "events", obj, t0, t1)` gives forward flow on the target and its `valid` mask, at 1280×720.
* **Metrics.** `oshd_bench.metrics.flow_stats` computes EPE, 1PE, 3PE and AE. `--mask valid` (default, used for Table III) uses all valid GT target pixels. `--mask valid_events` uses valid pixels with at least one event in `[t0, t1)`, and writes `<domain>_<split>_evmask.parquet`.
* **Output.** One row per sample in `results/flow/eraft/<setting>/<domain>_<split>.parquet`, with columns `trajectory, t_ns, t0_ns, object, n, sum_epe, n_gt1, n_gt3, sum_ae_deg, runtime_ms`. A `.json` file holds the summary and hardware. `oshd-bench eval` pools these into Table III (real_test).
* **runtime_ms** covers both GPU voxel grids, E-RAFT with 12 iterations, and upsampling. Events are already in memory, and GT rendering is excluded.

## Fine-tuning (ours; upstream has no training code)

`src/eraft_oshd/train.py` follows RAFT's fine-tuning recipe:

* **Loss.** Sequence loss `Σ_i 0.8^(N−i) · mean_valid |f_i − f_gt|₁` over the 12 iterative predictions, restricted to valid GT pixels (the target) with |f| < 400.
* **Optimiser.** AdamW with lr 1e-4, weight decay 1e-5 and eps 1e-8. OneCycle schedule (pct_start 0.05, linear anneal, no momentum cycling). Gradients are clipped at 1.0 and BatchNorm is frozen.
* **Batch and resolution.** Batch 4 at the inference resolution. The GT is rendered directly at 640×360 with `intrinsics.resized`, so no flow resampling is needed. It is cached sparsely in `work/eraft_cache/`.
* **Augmentation.** Random horizontal flips (p 0.5) and vertical flips (p 0.1), with the flow sign flipped accordingly.
* **Defaults.** 5000 steps.

## Commands

```bash
cd baselines/eraft && ./setup.sh
cd ../.. && export OSH_DATASET_ROOT=... OSHD_BENCH_RESULTS=...   # omit both for the real benchmark
cd baselines/eraft
flock ../../work/gpu.lock uv run eraft-predict --setting ZS --split test --domain real --debug 4
flock ../../work/gpu.lock uv run eraft-train  --steps 5000             # -> weights/trained/FT.tar
flock ../../work/gpu.lock uv run eraft-predict --setting FT --split test --domain real --debug 4
flock ../../work/gpu.lock uv run eraft-predict --setting FT --mask valid_events   # optional variant
cd ../.. && uv run --project bench oshd-bench eval                     # tables/table3_flow.*
```

`--limit N` evaluates N evenly spaced samples. `--debug N` saves colour-wheel panels to `work/debug/events/eraft/`: the voxel grid, the predicted flow, the prediction on the GT mask, the GT flow and the EPE.
