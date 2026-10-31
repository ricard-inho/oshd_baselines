# geotransformer: GeoTransformer on cropped lidar scans vs the CAD

This baseline fills Table I, LiDAR row, method `geotransformer`, settings **ZS** and **FT**. The modality and sensor are both `lidar`. The output pose is `T_lidar_model` in the CubeSat_U1 CAD frame.

| | |
|---|---|
| Upstream | [qinzheng93/GeoTransformer](https://github.com/qinzheng93/GeoTransformer) @ `e7a135af4c318ff3b8d7f6c963df094d7e4ea540`, MIT licence |
| Weights | `geotransformer-modelnet.pth.tar`, 21 MB, from the [1.0.0 release](https://github.com/qinzheng93/GeoTransformer/releases/tag/1.0.0) (MIT, same repo). Downloaded by `setup.sh` into `weights/`. |
| Env | uv project (torch 2.4.1+cu121, Python 3.12). **Not Docker**: upstream's `setup.py` declares a CUDAExtension, but every source is CPU C++ (grid subsampling, radius neighbours). `build_ext.py` compiles the same sources with torch's JIT `CppExtension` loader, so no nvcc is needed. The system nvcc 11.5 would not match torch's CUDA 12.1 anyway. |
| Crop | the training-free crop from `baselines/lidar_registration` (`oshd_lidar.crop.TargetCropper`, path dependency `oshd-lidar`). See that README. `runtime_ms` includes it. |

### Why the ModelNet weights, not 3DMatch or KITTI

Our problem is object-level: a partial, noisy view of a small object against a full CAD sample. That is exactly the ModelNet/RPMNet protocol GeoTransformer's ModelNet model was trained on (objects in the unit sphere, partial-to-partial, rotations up to 45°).

The 3DMatch model works on scene fragments with a 2.5 cm first voxel. Its coarse superpoints are about 20 cm apart, so the 10 cm cube would get at most one or two superpoints. The KITTI model uses a 30 cm voxel.

We therefore scale both clouds ×10: the cube becomes side 1, like a ModelNet object. The cube's 24-fold symmetry keeps the canonical target rotation within 63° of identity (see below), close to ModelNet's ≤ 45° training range.

## ZS / FT

* **ZS**: the ModelNet weights fine-tuned on train / `synthetic` pairs. If there are no synthetic train samples, ZS is the ModelNet weights unchanged ("pretrained only"), and `geotr-train --setting ZS` just copies them.
* **FT**: the ZS weights further fine-tuned on train / `real` pairs.

Training pairs have two sides:
* **Reference**: the cropped scan, centred on the crop centroid, ×10. It comes from the same auto crop as at test time. Where that crop misses (centroid more than 8 cm from GT), a GT-centred 10 cm sphere is used instead; this is training only.
* **Source**: 717 CAD body points, holder dropped, centred, ×10.

The ground-truth transform is the symmetric variant whose rotation is closest to identity. This is deterministic given the geometry, and the metrics are symmetry-aware. Augmentation applies a full random SO(3) rotation of the scene about the crop centroid, 5 mm jitter, 0–30 % point dropout, and a resampled CAD cloud.

The trainer is upstream's `IterBasedTrainer` with the ModelNet `model.py` / `loss.py`, following their `trainval.py`. It uses Adam at 1e-4 with warm-up and cosine decay, and holds out 10 % of the pairs for validation. Weights are loaded with `load_state_dict`; `--snapshot` would also restore the release's iteration counter of 400k.

## Commands

```bash
cd baselines/geotransformer && ./setup.sh && cd ../..
# (dev) export OSHD_DATASET_ROOT=$PWD/work/dev_dataset OSHD_BENCH_RESULTS=$PWD/work/dev_results OSHD_BENCHMARK=none
L=work/gpu.lock
flock $L uv run --project baselines/geotransformer geotr-train --setting ZS --iters 2000   # -> weights/oshd-ZS.pth.tar
flock $L uv run --project baselines/geotransformer geotr-train --setting FT --iters 1000   # -> weights/oshd-FT.pth.tar
for S in ZS FT; do for D in real synthetic; do
  flock $L uv run --project baselines/geotransformer geotr-predict --setting $S --domain $D --split test
done; done
uv run --project bench oshd-bench eval
# diagnosis only (GT-centred crop, setting <S>-oracle)
flock $L uv run --project baselines/geotransformer geotr-predict --setting ZS --crop oracle
uv run --project baselines/lidar_registration lidar-viz --method geotransformer --setting ZS
```

Training pairs are cached in `work/geotransformer/pairs/`. Delete them after the dataset changes. Logs and TensorBoard files go to `work/geotransformer/<setting>/`.

## Dev smoke results (fitted dev lidar extrinsic, `work/lidar_dev`, real test 009, 323 scans)

These runs used the dev data: 341 "synthetic" pairs from trajectory 008 and 362 real pairs from 007. ZS was 2000 iterations (102 s) and FT 1000 iterations (51 s).

| setting | e_t | e_q | ADD(-S) AUC | centre err. median | coverage | rate |
|---|---|---|---|---|---|---|
| ZS | 5.1 cm | 25.5° | 0.54 | 3.0 cm | 99.7 % | 31 Hz |
| FT | 5.4 cm | 26.0° | 0.53 | 3.1 cm | 99.7 % | 31 Hz |
| ZS-oracle crop | – | – | – | 2.5 cm | 100 % | – |

With this little data, fine-tuning does not help: validation RRE is about 47–59° and registration recall is 0 at upstream's thresholds. The numbers are dev-only and only show that the pipeline runs end to end. On `work/dev_dataset` with the placeholder extrinsic, which is about 180° off, `oshd-bench eval` fills the rows, but the errors are about 50 cm because the GT itself is wrong there.
