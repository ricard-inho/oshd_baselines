"""Train SPNv2 on OSHD.

* ``--setting ZS``: from ImageNet EfficientNet-B3 (or ``--init`` a SPEED+ checkpoint), on
  split ``train`` / domain ``synthetic`` only.
* ``--setting FT``: starts from the ZS checkpoint and fine-tunes on ``train`` / ``real``.

Architecture, losses, augmentations, optimiser, AMP and the per-epoch loop are upstream's
(``nets.build_spnv2``, ``engine.trainer.do_train``, ``solver.*``, ``dataset.transforms``).
Only the data comes from ``spnv2_oshd.convert`` (run automatically).
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
    UPSTREAM_COMMIT,
    VirtualCamera,
    build_cfg,
    ensure_upstream_on_path,
    run_dir,
)

log = logging.getLogger("spnv2_oshd.train")

DEFAULTS = {  # setting -> (epochs, lr, lr_steps)
    "ZS": (20, 1.0e-3, [15, 18]),   # upstream's schedule
    "FT": (10, 1.0e-4, [8]),
}


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

    class OshdSPEEDPlusDataset(SPEEDPLUSDataset):
        """Upstream dataset; only the hard-coded 768x512 folder names are derived from the size
        (the ``TODO`` in upstream's __init__)."""

        def __init__(self, cfg, split="train", transforms=None, target_generators=None):
            super().__init__(cfg, split, transforms, target_generators)
            w, h = cfg.DATASET.INPUT_SIZE
            s = cfg.DATASET.OUTPUT_SIZE[0]
            self.imagefolder = f"images_{w}x{h}_RGB"
            self.maskfolder = f"masks_{w // s}x{h // s}"

    return OshdSPEEDPlusDataset


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
    log.info(f"{what}: loaded {len(ok)}/{len(own)} tensors ({len(state) - len(ok)} skipped: shape/name mismatch)")


def train(sensor: str, setting: str, epochs: int | None = None, max_iters: int | None = None,
          batch: int | None = None, lr: float | None = None, init: str | None = None,
          stride: int = 1, limit: int | None = None, workers: int = 8, out: Path | None = None,
          zs_checkpoint: Path | None = None, opts: list | None = None) -> Path:
    import torch

    from .convert import convert, data_root, sensor_folder

    domain = "synthetic" if setting == "ZS" else "real"
    out = Path(out or run_dir(sensor, setting))
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(out / "train.log")], force=True)

    parent = None
    if setting == "FT":
        zs = Path(zs_checkpoint or run_dir(sensor, "ZS") / "model.pth.tar")
        if not zs.exists():
            raise SystemExit(f"FT starts from the ZS checkpoint, not found: {zs}")
        parent = torch.load(zs, map_location="cpu", weights_only=False)
        # the FT data must use the ZS virtual camera and symmetry reference
        sf = sensor_folder(sensor)
        sf.mkdir(parents=True, exist_ok=True)
        (sf / "vcam.json").write_text(json.dumps(parent["vcam"], indent=2))
        prev = sf / "canon.json"
        canon = json.loads(prev.read_text()) if prev.exists() else {}
        canon.update({"R_ref": parent["R_ref"], "center": parent["center"], "keypoints_net": parent["keypoints_net"]})
        prev.write_text(json.dumps(canon, indent=2))

    csv_rel = convert(sensor, domain, "train", stride=stride, limit=limit, workers=workers,
                      gray=parent["gray"] if parent else True)
    vcam = VirtualCamera(**json.loads((sensor_folder(sensor) / "vcam.json").read_text()))
    canon = json.loads((sensor_folder(sensor) / "canon.json").read_text())

    e_def, lr_def, steps = DEFAULTS[setting]
    ep = epochs or e_def
    o = ["TRAIN.TRAIN_CSV", str(csv_rel), "TRAIN.VAL_CSV", str(csv_rel), "TRAIN.END_EPOCH", ep,
         "TRAIN.LR", lr or lr_def, "TRAIN.LR_STEP", [s for s in steps if s < ep] or [ep]]
    if batch:
        o += ["TRAIN.IMAGES_PER_GPU", batch]
    cfg = build_cfg(sensor, vcam, data_root(), opts=o + list(opts or []))
    (out / "config.yaml").write_text(cfg.dump())

    from engine.trainer import do_train
    from nets import build_spnv2
    from solver import adjust_learning_rate, get_optimizer, get_scaler
    from utils.utils import set_seeds_cudnn

    set_seeds_cudnn(cfg, seed=cfg.SEED)
    model = build_spnv2(cfg)  # ImageNet EfficientNet-B3 backbone (torchvision weights)
    if parent is not None:
        model.load_state_dict(parent["state_dict"], strict=True)
        log.info("initialised from the ZS checkpoint")
    elif init:
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
    if max_iters:
        ep = min(ep, math.ceil(max_iters / n_per_epoch))
    log.info(f"{sensor}/{setting}: {len(loader.dataset)} samples, {n_per_epoch} it/epoch, {ep} epochs, "
             f"max_iters={max_iters}")
    done, t0 = 0, time.time()
    for epoch in range(ep):
        adjust_learning_rate(optimizer, epoch, cfg)
        n = n_per_epoch if not max_iters else min(n_per_epoch, max_iters - done)
        if n <= 0:
            break
        do_train(epoch, cfg, model, _Capped(loader, n), optimizer, device=device, scaler=scaler)
        done += n
        save(out, model, cfg, vcam, canon, sensor, setting, epoch + 1, done, parent)
    log.info(f"done: {done} iterations in {time.time() - t0:.0f} s -> {out / 'model.pth.tar'}")
    return out / "model.pth.tar"


def save(out, model, cfg, vcam, canon, sensor, setting, epoch, iters, parent) -> None:
    import torch

    torch.save({
        "state_dict": model.state_dict(), "cfg": cfg.dump(), "vcam": vcam.to_dict(),
        "R_ref": canon["R_ref"], "center": canon["center"], "keypoints_net": canon["keypoints_net"],
        "gray": True if parent is None else parent["gray"], "sensor": sensor, "setting": setting,
        "epoch": epoch, "iterations": iters, "upstream_commit": UPSTREAM_COMMIT,
        "parent_iterations": None if parent is None else parent["iterations"],
    }, out / "model.pth.tar")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sensor", required=True, choices=["firefly", "realsense_color"])
    ap.add_argument("--setting", required=True, choices=["ZS", "FT"])
    ap.add_argument("--epochs", type=int, default=None, help="default ZS 20, FT 10")
    ap.add_argument("--max-iters", type=int, default=None, help="stop after this many iterations (smoke tests)")
    ap.add_argument("--batch", type=int, default=None, help="images per batch (default 4)")
    ap.add_argument("--lr", type=float, default=None, help="default ZS 1e-3, FT 1e-4")
    ap.add_argument("--init", default=None, help="ZS only: SPEED+ checkpoint to initialise from (shape-matched)")
    ap.add_argument("--zs-checkpoint", default=None, help="FT only: default runs/<sensor>/ZS/model.pth.tar")
    ap.add_argument("--stride", type=int, default=1, help="use every n-th training frame")
    ap.add_argument("--limit", type=int, default=None, help="use the first n training frames")
    ap.add_argument("--workers", type=int, default=8, help="conversion processes")
    ap.add_argument("--out", default=None)
    ap.add_argument("opts", nargs=argparse.REMAINDER, help="extra yacs KEY VALUE overrides")
    a = ap.parse_args(argv)
    train(a.sensor, a.setting, a.epochs, a.max_iters, a.batch, a.lr, a.init, a.stride, a.limit, a.workers,
          a.out, a.zs_checkpoint, a.opts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
