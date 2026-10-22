# MegaPose baseline (RGB Firefly, RGB RealSense, E2VID + MegaPose)

MegaPose is a CAD-based, novel-object 6D pose estimator. It needs no training on the target: you give it an RGB image, the camera intrinsics, a 2D box and the CAD mesh (`models/CubeSat_U1.obj`, metres) and it returns `T_cam_model` in the CAD frame.

| Table I row | `--modality` | sensor frame | method key | input image |
|---|---|---|---|---|
| RGB (Firefly) | `rgb_firefly` | `firefly` | `megapose` | 3072x2048, undistorted and resized to 1152x768 |
| RGB (RealSense) | `rgb_realsense` | `realsense_color` | `megapose` | 640x480, undistorted |
| Events | `events` | `events` | `e2vid_megapose` | E2VID grayscale frame (1280x720, repeated to 3 channels), undistorted |

## Upstream, licence, weights

* **Code:** [agimus-project/happypose](https://github.com/agimus-project/happypose), the maintained MegaPose/CosyPose implementation, pinned at `6278e12b9ada314a5def8041d9f33c649fe82f40` (2026-01-16). happypose is BSD-2-Clause. The MegaPose sources inside it keep their original Apache-2.0 headers (Inria & NVIDIA, from megapose6d).
* **Weights:** the official MegaPose 1.0 checkpoints from the authors' Inria mirror, `https://www.paris.inria.fr/archive_ylabbeprojectsdata/megapose/megapose-models/`. This is the same mirror `happypose.toolbox.utils.download --megapose_models` uses. Only the final `checkpoint.pth.tar` and `config.yaml` of two runs are downloaded, about 170 MB, and `setup.sh` checks their sha256:
  * coarse model `coarse-rgb-906902141`
  * refiner `refiner-rgb-653307694`

  Together these are happypose's named model **`megapose-1.0-RGB`**. They were checked downloadable on 2026-09-28. No separate weights licence is published: they are released alongside the Apache-2.0 megapose6d code.
* **Settings used:**
  * the full 576-rotation coarse SO(3) grid,
  * 1 coarse hypothesis,
  * **5 refiner iterations**,
  * final pose chosen by the coarse model's scoring pass,
  * the Panda3D renderer (headless EGL).

  Each can be changed with `--so3-stride`, `--hypotheses` and `--refiner-iterations`, and the values are recorded in each prediction's `.json`.

## ZS / FT

MegaPose itself is **not re-trained** (it is training-free). ZS/FT only changes the 2D detector that initialises it, as the root README defines:

* **ZS** reads the Mask R-CNN detections trained on `train/synthetic` (baselines/detector).
* **FT** reads the Mask R-CNN detections fine-tuned on `train/real`.

The detections come from `detection_path(results, <sensor>, "maskrcnn", <setting>, <domain>, <split>)`. For the events row, the detections are on the E2VID frames under sensor `events`. `megapose-train` is a no-op that only checks the weights.

With `--detector oracle`, MegaPose uses the GT boxes and masks from `pooh-bench oracle-detections` (detector `oracle`, setting `GT`). Those results are written under setting **`ORACLE`** so they never fill the ZS/FT cells.

**Per-frame only.** Every sample is estimated independently, with no tracking or temporal prior.

## Details

* **Camera handling:**
  * Images are undistorted, and optionally resized, with `pooh_dataset.imaging.CameraView`. MegaPose gets the resulting pinhole K.
  * The detector's mask, or its filled box when there is no mask, is warped with the same map and gives the box in the MegaPose image.
  * The returned pose is in the sensor optical frame (undistorting and resizing only change the intrinsics), so it is written unchanged.
* **Multiple detections:** only the best-scoring detection per sample is used by default. With `--max-dets k`, MegaPose runs on the top k and keeps the pose with the highest MegaPose score.
* **`runtime_ms`:** the sum of
  * image decode, undistortion and resizing,
  * MegaPose (coarse + refiner + scoring, CUDA-synchronised),
  * the detection's `runtime_ms`,
  * for events, the E2VID `runtime_ms` from the frame index, if it has one.

  On the RTX 4090 with the full grid it is about 1.1–1.2 s per RealSense frame, and the coarse stage takes about 90% of that. `--so3-stride 8` is the upstream example's speed-up.
* **E2VID frames:** read from `<results>/e2vid/<domain>_<split>/<trajectory>/<t_ns>.png`, or from `--frames-dir <root>`. Per-frame reconstruction times are read from `<root>/<domain>_<split>.parquet` or `<root>/<domain>_<split>/index.parquet` when either has a `runtime_ms` column.
* **Ground truth:** inference reads only the `trajectory, domain, sensor, index, t_ns, object` columns of the GT sample list, never `T`. Only the `megapose-overlay` debug tool reads GT, to draw it in red.

## Commands

```bash
cd baselines/megapose && ./setup.sh
# production (real dataset & results/); run from anywhere inside the repo
for s in ZS FT; do for d in real synthetic; do   # synthetic test feeds S_sim (ZS only is reported)
  flock ../../work/gpu.lock uv run megapose-predict --modality rgb_firefly   --setting $s --split test --domain $d
  flock ../../work/gpu.lock uv run megapose-predict --modality rgb_realsense --setting $s --split test --domain $d
  flock ../../work/gpu.lock uv run megapose-predict --modality events        --setting $s --split test --domain $d
done; done
uv run --project ../../bench pooh-bench eval
```

Outputs go to `results/predictions/{rgb_firefly,rgb_realsense}/megapose/<setting>/<domain>_test.parquet` and `results/predictions/events/e2vid_megapose/<setting>/...`. When the synthetic test split or its detections don't exist yet, `predict` fails with a message naming the missing file.

### Smoke test on the dev dataset

```bash
cd /home/ricard/pooh_baselines
export POOH_DATASET_ROOT=$PWD/work/dev_dataset POOH_BENCH_RESULTS=$PWD/work/dev_results POOH_BENCHMARK=none
for s in realsense_color firefly events; do uv run --project bench pooh-bench oracle-detections --sensor $s; done
cd baselines/megapose
uv run megapose-events-standin --out ../../work/megapose_e2vid_standin --limit 20   # event-count images, E2VID layout
flock ../../work/gpu.lock uv run megapose-predict --modality rgb_realsense --detector oracle --limit 20
flock ../../work/gpu.lock uv run megapose-predict --modality rgb_firefly   --detector oracle --limit 20
flock ../../work/gpu.lock uv run megapose-predict --modality events --detector oracle --limit 20 \
      --frames-dir ../../work/megapose_e2vid_standin
uv run megapose-overlay --modality rgb_realsense --setting ORACLE --n 6      # -> work/debug/megapose/*.jpg
uv run --project ../../bench pooh-bench eval   # needs a ZS/FT file to show up in table1 (see --out-setting)
```

To check that `pooh-bench eval` fills the Table I cells without a trained detector, add `--out-setting ZS` to an oracle run in the **dev** results dir. Never do this in `results/`.

## Notes

* happypose's Panda3D renderer starts worker processes that can keep a finished Python process alive, and with it the `flock work/gpu.lock`. `megapose-predict` therefore kills its workers and exits hard. Any ad-hoc script that builds a `MegaPose` object must do the same.
* Dev smoke results (2026-09-28, `--limit 20`, dev dataset):
  * RealSense with oracle boxes: the overlays align with the cube to about 2–3 cm.
  * ZS with the dev Mask R-CNN: e_t 2.4 cm, e_q 3.3°.
  * Firefly and events: the dev extrinsics are placeholders, so their numbers are meaningless.
