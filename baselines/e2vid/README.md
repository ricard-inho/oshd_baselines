# E2VID: event -> frame reconstruction

Reconstructs grayscale intensity frames from the EVK4 event stream so that frame-based
methods can run on events. It feeds the **E2VID + MegaPose** row of Table I, and it gives
the detector its event-camera input images (`baselines/detector`, `--sensor events`).
It has no pose output and no ZS/FT setting of its own. The public weights are used as they
are, with no retraining. For the E2VID + MegaPose row, ZS/FT is the detector's training.

| | |
|---|---|
| Upstream | https://github.com/uzh-rpg/rpg_e2vid, commit `d0a7c005f460f2422f2a4bf605f70820ea7a1e5f` (2019-12-29), cloned into `third_party/` by `setup.sh` (not vendored) |
| Licence | GPL-3.0 (upstream code). Our wrapper `src/e2vid_pooh/` only imports it at runtime |
| Weights | `E2VID_lightweight.pth.tar` from the upstream README: https://rpg.ifi.uzh.ch/data/E2VID/models/E2VID_lightweight.pth.tar (sha256 `4cfeb2c850bf48fc9fa907e969cb8a04e3c51314da2d65bdb81145ac96574128`), saved to `weights/` |
| Paper | Rebecq et al., "High Speed and High Dynamic Range Video with an Event Camera", TPAMI 2019 |
| Env | uv project (Python 3.12, torch 2.6.0+cu124) |

## What it does

For every trajectory in a GT sample list (`results/gt/events/<domain>_<split>.parquet`), the
event stream runs through the recurrent network **causally**: it starts at the first event,
goes in time order, and never looks ahead.

* `--mode count` (default, the upstream default). Windows hold a fixed number of events,
  N = 0.35 · W · H = 322 560. Only complete windows advance the recurrent state. For a sample
  time t, one extra forward pass on the unfinished window `[last window end, t)` produces
  the frame, and the state that pass returns is discarded. So the frame at t uses exactly
  the events before t.
* `--mode duration` (upstream `--fixed_duration`). Uses 50 ms windows aligned to the 20 Hz
  sample grid.

Post-processing matches the upstream defaults: event-tensor normalisation, unsharp mask
0.3 / σ 1, fixed Imin = 0 and Imax = 1 (no auto-HDR), and no bilateral filter.

## Output (the contract)

```
results/e2vid/<domain>_<split>/<trajectory>/<t_ns>.png  uint8 grayscale 1280x720, raw event-sensor pixels (NOT undistorted)
results/e2vid/<domain>_<split>/index.parquet           trajectory, t_ns, path (relative to results/), runtime_ms
results/e2vid/<domain>_<split>/index.json              settings, weights hash, hardware
```

One PNG is written per unique `(trajectory, t_ns)` of the sample list. `runtime_ms` is the
amortised cost of a 20 Hz streaming reconstructor. It is the GPU and CPU time (voxel grid,
network and post-processing, but not disk I/O) spent between the previous 20 Hz grid point
and this sample. That includes the sample's partial-window pass. The median on an RTX 4090
is about 22 ms. A pose method on events adds this to its own runtime.

The sample lists' `T` column is never read.

## Commands

```bash
cd /home/ricard/pooh_baselines
./baselines/e2vid/setup.sh
# dev: export POOH_DATASET_ROOT=$PWD/work/dev_dataset POOH_BENCH_RESULTS=$PWD/work/dev_results
cd baselines/e2vid
flock ../../work/gpu.lock uv run python -m e2vid_pooh.reconstruct --all          # every gt/events/*.parquet (train + test)
flock ../../work/gpu.lock uv run python -m e2vid_pooh.reconstruct --domain real --split test [--limit N]
```

Options: `--mode count|duration`, `--num-events-per-pixel 0.35`, `--window-events N`,
`--trajectory NAME`, `--cpu`.

A full trajectory (about 40 s, 24 to 127 M events) takes under a minute on the GPU. Each frame is a PNG of about 0.3 to 0.6 MB.
