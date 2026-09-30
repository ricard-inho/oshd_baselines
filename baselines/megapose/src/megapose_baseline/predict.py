"""MegaPose predictions for the POOH-D Table I rows (per-frame, no temporal tracking).

    megapose-predict --modality rgb_realsense --setting ZS --split test --domain real [--limit N]

Initialised by 2D detections: ``--detector maskrcnn`` (default; the ZS / FT detector of
baselines/detector) or ``--detector oracle`` (GT boxes; output setting defaults to ORACLE).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .common import MODALITIES, CameraPrep, E2VIDFrames, group_detections, load_detections, load_samples


def parse(argv=None):
    ap = argparse.ArgumentParser(prog="megapose-predict", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modality", required=True, choices=sorted(MODALITIES))
    ap.add_argument("--setting", default="ZS", help="ZS | FT: which detector training initialises MegaPose")
    ap.add_argument("--split", default="test")
    ap.add_argument("--domain", default="real", choices=["real", "synthetic"])
    ap.add_argument("--detector", default="maskrcnn", choices=["maskrcnn", "oracle"])
    ap.add_argument("--out-setting", default=None,
                    help="setting folder to write (default: --setting, or ORACLE with --detector oracle)")
    ap.add_argument("--frames-dir", default=None,
                    help="events only: E2VID frames root (default <results>/e2vid)")
    ap.add_argument("--size", default=None, help="MegaPose input WxH (default: 1152x768 firefly, native otherwise)")
    ap.add_argument("--refiner-iterations", type=int, default=5)
    ap.add_argument("--hypotheses", type=int, default=1, help="coarse hypotheses refined per detection")
    ap.add_argument("--so3-stride", type=int, default=1, help="subsample the 576-rotation coarse grid")
    ap.add_argument("--max-dets", type=int, default=1, help="detections per frame tried (best pose score kept)")
    ap.add_argument("--min-det-score", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=None, help="first N samples (smoke tests)")
    ap.add_argument("--stride", type=int, default=1, help="every k-th sample")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    a = parse(argv)
    import pooh_bench
    from pooh_bench import PoseWriter, prediction_path

    from .megapose import MegaPose

    m = MODALITIES[a.modality]
    sensor, method = m["sensor"], m["method"]
    out_setting = a.out_setting or ("ORACLE" if a.detector == "oracle" else a.setting)
    size = tuple(int(v) for v in a.size.lower().split("x")) if a.size else m["size"]

    ds = pooh_bench.open_dataset()
    results = pooh_bench.results_dir()
    samples = load_samples(results, sensor, a.domain, a.split).iloc[::a.stride]
    if a.limit:
        samples = samples.iloc[:a.limit]
    det_df, det_path = load_detections(results, sensor, a.detector, a.setting, a.domain, a.split)
    dets = group_detections(det_df[det_df["score"] >= a.min_det_score], a.max_dets)
    frames = None
    if sensor == "events":
        frames = E2VIDFrames(Path(a.frames_dir) if a.frames_dir else results / "e2vid", a.domain, a.split)

    objs = sorted(set(samples["object"]))
    models = {o: MegaPose(o, ds.objects[o].mesh_path, a.refiner_iterations, a.hypotheses, a.so3_stride)
              for o in objs}
    path = prediction_path(results, a.modality, method, out_setting, a.domain, a.split)
    w = PoseWriter(path, {
        "method": method, "modality": a.modality, "sensor": sensor, "setting": out_setting,
        "detector": a.detector, "detections": str(det_path), "input_size": size,
        "temporal": "none (per-frame estimation)", "frames_dir": str(frames.dir) if frames else None,
        "megapose": next(iter(models.values())).info if models else None,
        "upstream": "agimus-project/happypose@6278e12b9ada314a5def8041d9f33c649fe82f40",
    })

    preps: dict = {}
    n_nodet = n_noimg = n_fail = 0
    for r in tqdm(samples.itertuples(index=False), total=len(samples), desc=f"megapose {a.modality}"):
        cands = dets.get((r.trajectory, int(r.t_ns), r.object), [])
        if not cands:
            n_nodet += 1
            continue
        traj = ds[r.trajectory]
        t0 = time.perf_counter()
        extra_ms = 0.0
        if frames is not None:
            img, extra_ms = frames.get(r.trajectory, int(r.t_ns))
            if img is None:
                n_noimg += 1
                continue
        else:
            img = traj.frame(sensor, int(r.index))
        key = (r.trajectory, sensor)
        if key not in preps:
            preps[key] = CameraPrep(traj.calibration.intrinsics(sensor), size)
        prep = preps[key]
        rgb = prep.image(img)
        boxes, used = [], []
        for d in cands:
            b = prep.box(d)
            if b is not None and b[2] - b[0] >= 2 and b[3] - b[1] >= 2:
                boxes.append(b)
                used.append(d)
        if not boxes:
            n_nodet += 1
            continue
        prep_ms = (time.perf_counter() - t0) * 1000
        try:
            res, mp_ms = models[r.object](rgb, prep.intr.K, boxes)
        except Exception as e:  # noqa: BLE001  (one bad frame must not kill a long run)
            n_fail += 1
            tqdm.write(f"{r.trajectory} {r.t_ns}: {type(e).__name__}: {e}")
            continue
        k = int(np.argmax([s if s is not None else -np.inf for s in (x[1] if x else None for x in res)]))
        if res[k] is None:
            n_fail += 1
            continue
        T, score = res[k]
        det_ms = float(used[k].runtime_ms) if np.isfinite(used[k].runtime_ms) else 0.0
        w.add(r.trajectory, sensor, int(r.t_ns), r.object, T, score, prep_ms + mp_ms + det_ms + extra_ms)
    w.meta.update(n_samples=len(samples), n_predicted=len(w), n_no_detection=n_nodet, n_no_frame=n_noimg,
                  n_failed=n_fail)
    print(f"{w.save()}: {len(w)}/{len(samples)} poses (no det {n_nodet}, no frame {n_noimg}, failed {n_fail})")
    return 0


def cli() -> None:
    """Entry point. Exits hard: happypose's Panda3D renderer worker processes can otherwise keep
    the interpreter (and the shared ``flock work/gpu.lock``) alive after the run has finished."""
    import os
    import sys

    try:
        code = main()
    except BaseException:
        import traceback

        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    import multiprocessing as mp

    for c in mp.active_children():  # renderer workers inherit the flock fd: stop them first
        c.kill()
    for c in mp.active_children():
        c.join(timeout=5)
    os._exit(code)


if __name__ == "__main__":
    cli()
