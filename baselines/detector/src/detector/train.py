"""Train the target detector.

    ZS: COCO init -> synthetic train split       (weights/<sensor>/ZS.pt)
    FT: ZS weights -> real train split           (weights/<sensor>/FT.pt)

Labels: for every sample of results/gt/<sensor>/<domain>_train.parquet (the train split is
the only place the ``T`` column is read) the CAD silhouette is rendered with the GT pose
through the (distorted) sensor intrinsics at the network input resolution; its tight box is
the box label. Samples whose silhouette has < ``--min-px`` pixels become negatives.
For ``--sensor events`` the input images are the E2VID reconstructions (baselines/e2vid).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .common import (
    SENSORS,
    _DS,
    build_model,
    checkpoint_path,
    gt_mask,
    load_image,
    object_names,
    sample_list,
    to_input,
)


class TrainSet(Dataset):
    def __init__(self, sensor: str, domain: str, split: str, limit: int | None, classes: list[str],
                 min_px: int = 20, augment: bool = True):
        self.sensor, self.domain, self.split = sensor, domain, split
        df = sample_list(sensor, domain, split, with_pose=True, limit=limit)
        self.groups = [g for _, g in df.groupby(["trajectory", "t_ns"], sort=True)]
        self.classes, self.min_px, self.augment = classes, min_px, augment

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, i):
        ds = _DS.get()
        g = self.groups[i]
        r = g.iloc[0]
        img = to_input(load_image(ds, self.sensor, self.domain, self.split, r.trajectory, r["index"], r.t_ns),
                       self.sensor)
        h, w = img.shape[:2]
        boxes, labels, masks = [], [], []
        for _, o in g.iterrows():
            if o.object not in self.classes:
                continue
            m = gt_mask(ds, self.sensor, o.trajectory, o.t_ns, o.object, o["T"], (w, h))
            if m.sum() < self.min_px:
                continue
            ys, xs = np.nonzero(m)
            boxes.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
            labels.append(self.classes.index(o.object) + 1)
            masks.append(m)
        x = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div_(255)
        t = {"boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
             "labels": torch.tensor(labels, dtype=torch.int64),
             "masks": torch.from_numpy(np.array(masks, np.uint8).reshape(-1, h, w))}
        if self.augment:
            x, t = augment(x, t, gray=self.sensor == "events")
        return x, t


def augment(x: torch.Tensor, t: dict, gray: bool):
    if random.random() < 0.5:  # horizontal flip
        x = x.flip(-1)
        W = x.shape[-1]
        b = t["boxes"].clone()
        t["boxes"] = torch.stack([W - b[:, 2], b[:, 1], W - b[:, 0], b[:, 3]], -1) if len(b) else b
        t["masks"] = t["masks"].flip(-1)
    # photometric: brightness / contrast / gamma (+ saturation for colour)
    x = x * random.uniform(0.6, 1.4)
    mean = x.mean()
    x = (x - mean) * random.uniform(0.7, 1.3) + mean
    x = x.clamp(0, 1) ** random.uniform(0.7, 1.4)
    if not gray:
        g = x.mean(0, keepdim=True)
        x = g + (x - g) * random.uniform(0.6, 1.4)
        if random.random() < 0.1:
            x = x.mean(0, keepdim=True).expand_as(x)
    if random.random() < 0.3:
        x = x + torch.randn_like(x) * random.uniform(0.0, 0.04)
    return x.clamp(0, 1).contiguous(), t


def collate(b):
    return tuple(zip(*b))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sensor", required=True, choices=SENSORS)
    ap.add_argument("--setting", required=True, choices=["ZS", "FT"])
    ap.add_argument("--domain", default=None, help="default: synthetic for ZS, real for FT")
    ap.add_argument("--split", default="train")
    ap.add_argument("--init", default=None, help="checkpoint to start from (default: COCO for ZS, ZS.pt for FT)")
    ap.add_argument("--epochs", type=float, default=None, help="default 12 (ZS) / 6 (FT)")
    ap.add_argument("--max-iters", type=int, default=None, help="stop after N iterations (smoke tests)")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=None, help="default 0.02 (ZS) / 0.005 (FT), for batch 4")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None, help="use the first N samples only")
    ap.add_argument("--min-px", type=int, default=20)
    ap.add_argument("--out", default=None, help="default weights/<sensor>/<setting>.pt")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    torch.manual_seed(a.seed)
    random.seed(a.seed)
    np.random.seed(a.seed)
    domain = a.domain or ("synthetic" if a.setting == "ZS" else "real")
    epochs = a.epochs if a.epochs is not None else (12 if a.setting == "ZS" else 6)
    lr = a.lr if a.lr is not None else (0.02 if a.setting == "ZS" else 0.005)
    classes = object_names()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    init = a.init or (None if a.setting == "ZS" else str(checkpoint_path(a.sensor, "ZS")))
    model = build_model(a.sensor, len(classes), coco_init=init is None)
    if init:
        ck = torch.load(init, map_location="cpu", weights_only=False)
        if ck["classes"] != classes:
            raise SystemExit(f"{init} was trained for {ck['classes']}, config has {classes}")
        model.load_state_dict(ck["model"])
        print(f"init from {init}")
    else:
        print("init from torchvision COCO (MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1)")
    model.to(device).train()

    data = TrainSet(a.sensor, domain, a.split, a.limit, classes, a.min_px)
    if not len(data):
        raise SystemExit(f"no {domain}/{a.split} samples for {a.sensor}")
    dl = DataLoader(data, batch_size=a.batch_size, shuffle=True, num_workers=a.workers, collate_fn=collate,
                    drop_last=len(data) >= a.batch_size, persistent_workers=a.workers > 0)
    total = int(math.ceil(epochs * len(dl)))
    if a.max_iters:
        total = min(total, a.max_iters)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=1e-4)
    warm = min(500, max(1, total // 10))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda it: min(1.0, (it + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(it, total) / total)))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    print(f"{a.sensor} {a.setting}: {len(data)} images ({domain}/{a.split}), {total} iterations, lr {lr}")

    it, t0, log = 0, time.time(), []
    while it < total:
        for imgs, tgts in dl:
            imgs = [x.to(device, non_blocking=True) for x in imgs]
            tgts = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in tgts]
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                losses = model(imgs, tgts)
            loss = sum(losses.values())
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 10.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            it += 1
            if it % 20 == 0 or it == total or it == 1:
                rec = {"iter": it, "loss": float(loss), **{k: float(v) for k, v in losses.items()},
                       "lr": opt.param_groups[0]["lr"], "elapsed_s": time.time() - t0}
                log.append(rec)
                print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()),
                      flush=True)
            if it >= total:
                break

    out = checkpoint_path(a.sensor, a.setting) if a.out is None else __import__("pathlib").Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = {"sensor": a.sensor, "setting": a.setting, "domain": domain, "split": a.split, "init": init or "coco",
            "iters": it, "epochs": epochs, "batch_size": a.batch_size, "lr": lr, "n_images": len(data),
            "limit": a.limit, "max_iters": a.max_iters}
    torch.save({"model": model.state_dict(), "classes": classes, "meta": meta}, out)
    out.with_suffix(".json").write_text(json.dumps({**meta, "log": log}, indent=2))
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
