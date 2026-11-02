"""Train SPNv2 on event time surfaces of the sim docking train list (satellite mockup only).

One setting, ``TS``: from ImageNet EfficientNet-B3 (or ``--init`` a SPEED+ / earlier
checkpoint, shape-matched), on the time surfaces of the ``satellite_mockup`` samples of an
``oshd-bench sim-prepare`` train list (default ``results/sim/sim_zerog_docking_train_v3``).

Architecture, losses, augmentations, optimiser, AMP and the per-epoch loop are upstream's
(``nets.build_spnv2``, ``engine.trainer.do_train``, ``solver.*``, ``dataset.transforms``).
Only the data comes from ``spnv2_events.convert`` (run automatically, idempotent).
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import math
import time
from pathlib import Path

from .common import (
    SETTING,
    UPSTREAM_COMMIT,
    TargetFrame,
    VirtualCamera,
    build_cfg,
    ensure_upstream_on_path,
    repo_root,
    run_dir,
)

log = logging.getLogger("spnv2_events.train")

TRAIN_RUN = "results/sim/sim_zerog_docking_train_v3"
EPOCHS, LR, LR_STEPS = 20, 1.0e-3, [15, 18]   # upstream's schedule


class _Capped:
    """A DataLoader limited to ``n`` batches (upstream's do_train iterates len(loader))."""

    def __init__(self, loader, n):
        self.loader, self.n = loader, n

    def __len__(self):
        return self.n

    def __iter__(self):
        return itertools.islice(iter(self.loader), self.n)


def make_dataset_class():
    ensure_upstream_on_path()
    from dataset.SPEEDPLUSDataset import SPEEDPLUSDataset

    class EventsSPEEDPlusDataset(SPEEDPLUSDataset):
        """Upstream dataset; only the hard-coded 768x512 folder names are derived from the size
        (the ``TODO`` in upstream's __init__): ``images_640x512_RGB``, ``masks_160x128``."""

        def __init__(self, cfg, split="train", transforms=None, target_generators=None):
            super().__init__(cfg, split, transforms, target_generators)
            w, h = cfg.DATASET.INPUT_SIZE
            s = cfg.DATASET.OUTPUT_SIZE[0]
            self.imagefolder = f"images_{w}x{h}_RGB"
            self.maskfolder = f"masks_{w // s}x{h // s}"

    return EventsSPEEDPlusDataset


def get_train_loader(cfg):
    """upstream dataset.build.get_dataloader(split='train'), with our dataset subclass."""
    import torch
    from dataset.build import _seed_worker
    from dataset.target_generators import HeatmapGenerator
    from dataset.transforms import build_transforms

    gens = [HeatmapGenerator([int(s / cfg.DATASET.OUTPUT_SIZE[0]) for s in cfg.DATASET.INPUT_SIZE],
                             cfg.DATASET.NUM_KEYPOINTS, cfg.DATASET.SIGMA) if h == "heatmap" else None
            for h in cfg.MODEL.HEAD.LOSS_HEADS]
    ds = make_dataset_class()(cfg, "train", build_transforms(cfg, is_train=True, load_labels=True), gens)
    return torch.utils.data.DataLoader(ds, batch_size=cfg.TRAIN.IMAGES_PER_GPU, shuffle=True,
                                       num_workers=min(cfg.TRAIN.IMAGES_PER_GPU, cfg.TRAIN.WORKERS),
                                       pin_memory=True, drop_last=True, worker_init_fn=_seed_worker,
                                       persistent_workers=False)


def load_partial(model, state: dict, what: str) -> None:
    own = model.state_dict()
    ok = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
    model.load_state_dict(ok, strict=False)
    log.info(f"{what}: loaded {len(ok)}/{len(own)} tensors "
             f"({len(state) - len(ok)} skipped: shape/name mismatch)")


def resolve(path) -> Path:
    """``path`` as given, or relative to the repo root if it does not exist from the cwd."""
    p = Path(path)
    return p if p.is_absolute() or p.exists() else repo_root() / p


def train(train_run: Path, setting: str = SETTING, epochs: int = EPOCHS, max_iters: int | None = None,
          batch: int | None = None, lr: float = LR, lr_steps: list | None = None, init: str | None = None,
          stride: int = 1, limit: int | None = None, workers: int = 8, out: Path | None = None,
          opts: list | None = None) -> Path:
    import torch

    from .convert import convert, data_root, sensor_folder

    train_run = resolve(train_run)
    out = Path(out or run_dir(setting))
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(out / "train.log")],
                        force=True)

    csv_rel = convert(train_run, stride=stride, limit=limit, workers=workers)
    sf = sensor_folder()
    vcam = VirtualCamera(**json.loads((sf / "vcam.json").read_text()))
    canon = json.loads((sf / "canon.json").read_text())
    tf = TargetFrame.from_dict(json.loads((sf / "target.json").read_text()))

    steps = [s for s in (lr_steps or LR_STEPS) if s < epochs] or [epochs]
    o = ["TRAIN.TRAIN_CSV", str(csv_rel), "TRAIN.VAL_CSV", str(csv_rel), "TRAIN.END_EPOCH", epochs,
         "TRAIN.LR", lr, "TRAIN.LR_STEP", steps]
    if batch:
        o += ["TRAIN.IMAGES_PER_GPU", batch]
    cfg = build_cfg(vcam, data_root(), opts=o + list(opts or []))
    (out / "config.yaml").write_text(cfg.dump())

    from engine.trainer import do_train
    from nets import build_spnv2
    from solver import adjust_learning_rate, get_optimizer, get_scaler
    from utils.utils import set_seeds_cudnn

    set_seeds_cudnn(cfg, seed=cfg.SEED)
    model = build_spnv2(cfg)  # ImageNet EfficientNet-B3 backbone (torchvision weights)
    if init:
        st = torch.load(init, map_location="cpu", weights_only=False)
        st = st.get("state_dict", st.get("best_state_dict", st))
        st = {k.removeprefix("module."): v for k, v in st.items()}
        load_partial(model, st, f"init {init}")
    device = torch.device("cuda")
    model = model.to(device)

    loader = get_train_loader(cfg)
    optimizer = get_optimizer(cfg, model)
    scaler = get_scaler(cfg)
    n_per_epoch = len(loader)
    ep = epochs
    if max_iters:
        ep = min(ep, math.ceil(max_iters / n_per_epoch))
    log.info(f"{setting}: {len(loader.dataset)} samples of {train_run.name}, "
             f"batch {cfg.TRAIN.IMAGES_PER_GPU}, "
             f"{n_per_epoch} it/epoch, {ep} epochs, lr {cfg.TRAIN.LR} steps {list(cfg.TRAIN.LR_STEP)}, "
             f"max_iters={max_iters}")
    done, t0, timing = 0, time.time(), []
    for epoch in range(ep):
        adjust_learning_rate(optimizer, epoch, cfg)
        n = n_per_epoch if not max_iters else min(n_per_epoch, max_iters - done)
        if n <= 0:
            break
        te = time.time()
        do_train(epoch, cfg, model, _Capped(loader, n), optimizer, device=device, scaler=scaler)
        dt = time.time() - te
        done += n
        timing.append({"epoch": epoch + 1, "iterations": n, "wall_s": round(dt, 1),
                       "it_per_s": round(n / dt, 3)})
        log.info(f"epoch {epoch + 1}/{ep}: {n} it in {dt:.0f} s "
                 f"({n / dt:.2f} it/s, {1000 * dt / n:.0f} ms/it), "
                 f"lr {optimizer.param_groups[0]['lr']:.1e}")
        save(out, model, cfg, vcam, canon, tf, setting, epoch + 1, done, train_run.name, timing)
    log.info(f"done: {done} iterations in {time.time() - t0:.0f} s -> {out / 'model.pth.tar'}")
    return out / "model.pth.tar"


def save(out, model, cfg, vcam, canon, tf, setting, epoch, iters, train_run, timing) -> None:
    import torch

    wall = sum(t["wall_s"] for t in timing)
    n = sum(t["iterations"] for t in timing)
    torch.save({
        "state_dict": model.state_dict(), "cfg": cfg.dump(), "vcam": vcam.to_dict(), "R_ref": canon["R_ref"],
        "target": tf.to_dict(), "setting": setting, "epoch": epoch, "iterations": iters,
        "upstream_commit": UPSTREAM_COMMIT, "train_run": train_run,
        "timing": {"epochs": timing, "wall_s": round(wall, 1),
                   "ms_per_it": round(1000 * wall / max(n, 1), 1)},
    }, out / "model.pth.tar")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", default=TRAIN_RUN,
                    help=f"oshd-bench sim-prepare train list (default {TRAIN_RUN})")
    ap.add_argument("--setting", default=SETTING, choices=[SETTING])
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--max-iters", type=int, default=None,
                    help="stop after this many iterations (smoke tests)")
    ap.add_argument("--batch", type=int, default=None, help="images per batch (default from the yaml, 4)")
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--lr-steps", type=int, nargs="+", default=LR_STEPS,
                    help="epochs where the lr drops x0.1")
    ap.add_argument("--init", default=None, help="checkpoint to initialise from (shape-matched partial load)")
    ap.add_argument("--stride", type=int, default=1, help="use every n-th sample of the list")
    ap.add_argument("--limit", type=int, default=None, help="use n evenly spaced samples (smoke tests)")
    ap.add_argument("--workers", type=int, default=8, help="conversion processes")
    ap.add_argument("--out", default=None, help="default <work>/runs/<setting>")
    ap.add_argument("opts", nargs=argparse.REMAINDER, help="extra yacs KEY VALUE overrides")
    a = ap.parse_args(argv)
    train(Path(a.run_dir), a.setting, a.epochs, a.max_iters, a.batch, a.lr, a.lr_steps, a.init, a.stride,
          a.limit, a.workers, a.out, a.opts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
