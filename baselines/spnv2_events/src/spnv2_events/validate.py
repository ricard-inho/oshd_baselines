"""Per-epoch validation on the held-out trajectories of the train list (``configs/val_trajectories.txt``).

The validation run-dir (``<work>/val/<train_run>_val/``) is a sim-run-format copy of the train
list restricted to the validation trajectories, evenly sub-sampled; a checkpoint is scored by
``predict.predict`` (reads ``samples.parquet`` only) and ``oshd_bench.sim.evaluate``.
"""

from __future__ import annotations

import contextlib
import ctypes
import gc
import io
import json
import logging
import shutil
from pathlib import Path

import pandas as pd

from .common import OBJECT, work_dir
from .convert import val_trajectories

SUBSET = f"object == '{OBJECT}'"
KEYS = ("speed_sym_mean", "e_t_cm_mean", "e_q_sym_deg_mean", "coverage", "n_samples")


def val_run_dir(train_run: Path, limit: int = 800) -> Path:
    """``<work>/val/<train_run>_val/`` (built once): samples / gt of the validation trajectories,
    at most ``limit`` rows spread over all of them, plus ``objects.json`` and ``prepare.json``."""
    train_run = Path(train_run)
    out = work_dir() / "val" / f"{train_run.name}_val"
    meta = {"limit": limit, "trajectories": sorted(val_trajectories())}
    if (out / "gt.parquet").exists() and json.loads((out / "val.json").read_text()) == meta:
        return out
    out.mkdir(parents=True, exist_ok=True)
    samples, gt = (pd.read_parquet(train_run / n) for n in ("samples.parquet", "gt.parquet"))
    key = ["trajectory", "t_ns", "object"]
    assert samples[key].equals(gt[key]), f"{train_run}: samples.parquet and gt.parquet rows differ"
    sel = gt[gt["trajectory"].isin(meta["trajectories"]) & (gt["object"] == OBJECT)].sort_values(["trajectory", "t_ns"])
    idx = sel.index[:: max(1, -(-len(sel) // limit))]
    samples.loc[idx].to_parquet(out / "samples.parquet", index=False)
    gt.loc[idx].to_parquet(out / "gt.parquet", index=False)
    for name in ("objects.json", "prepare.json"):
        shutil.copy(train_run / name, out / name)
    (out / "val.json").write_text(json.dumps(meta, indent=2))
    return out


def reset_thread_locale() -> None:
    """Back to the process locale (glibc ``uselocale(LC_GLOBAL_LOCALE)``): the network's first
    CUDA calls (training, inference) leave the main thread on the "C" locale, so files written with
    the default encoding (``sim.evaluate``'s RESULTS markdown, with a degree sign) fail as ASCII."""
    ctypes.CDLL(None).uselocale(ctypes.c_void_p(-1))


def validate(val_dir: Path, checkpoint: Path, out: Path, epoch: int) -> dict:
    """Predict + score ``checkpoint`` on ``val_dir``; ``groups['all']`` metrics (``KEYS``).
    Predictions and metrics go to ``out/predictions_epNN.parquet`` / ``out/metrics_epNN.*``."""
    import torch
    from oshd_bench.sim import evaluate

    from .predict import predict

    out.mkdir(parents=True, exist_ok=True)
    pred = out / f"predictions_ep{epoch:02d}.parquet"
    logging.disable(logging.INFO)  # model build / tqdm / results table: not for the train log
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            predict(val_dir, checkpoint=checkpoint, overlays=0, out=pred)
            reset_thread_locale()
            m = evaluate(val_dir, pred, subset=SUBSET, name=f"ep{epoch:02d}")
    finally:
        logging.disable(logging.NOTSET)
    gc.collect()
    torch.cuda.empty_cache()
    return {k: float(m["groups"]["all"][k]) for k in KEYS}
