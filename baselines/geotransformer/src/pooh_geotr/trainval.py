"""Fine-tune GeoTransformer (ModelNet weights) on cube pairs, following upstream's trainval.py.

    geotr-train --setting ZS [--iters 2000]    # init ModelNet weights, train on train/synthetic
    geotr-train --setting FT [--iters 1000]    # init pooh-ZS weights, train on train/real

The output is ``weights/pooh-<setting>.pth.tar``. Logs and snapshots go to
``work/geotransformer/<setting>/``. A random 10% of the pairs is held out for validation
(there is no val split). If the synthetic train split has no samples, ZS copies the ModelNet
weights unchanged ("pretrained only").
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time

import numpy as np
import torch


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--setting", required=True, choices=["ZS", "FT"])
    ap.add_argument("--iters", type=int, default=None, help="default: 2000 (ZS) / 1000 (FT)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--init", default=None, help="init weights (default: ModelNet for ZS, pooh-ZS for FT)")
    a, rest = ap.parse_known_args()
    sys.argv = [sys.argv[0]] + rest  # upstream trainer parses its own flags (--log_steps ...)

    import pooh_bench
    from pooh_lidar.common import OBJECT

    from .common import PRETRAINED, CubePairDataset, build_pairs, import_upstream, make_cfg, run_model, weights_for

    domain = "synthetic" if a.setting == "ZS" else "real"
    init = a.init or (PRETRAINED if a.setting == "ZS" else weights_for("ZS"))
    iters = a.iters or (2000 if a.setting == "ZS" else 1000)
    try:
        pairs = build_pairs(domain, "train")
    except FileNotFoundError:
        pairs = {"points": np.zeros(0, object), "T": np.zeros((0, 4, 4))}
    out = weights_for(a.setting)
    out.parent.mkdir(parents=True, exist_ok=True)
    if len(pairs["points"]) == 0:
        print(f"no {domain}/train pairs: {a.setting} = init weights {init} unchanged")
        shutil.copy(init, out)
        return

    cfg = make_cfg(a.setting, iters, a.lr)
    _, gt_model, gt_loss = import_upstream()
    from geotransformer.engine.iter_based_trainer import IterBasedTrainer
    from geotransformer.utils.data import (
        build_dataloader_stack_mode,
        calibrate_neighbors_stack_mode,
        registration_collate_fn_stack_mode,
    )
    from geotransformer.utils.torch import build_warmup_cosine_lr_scheduler

    obj = pooh_bench.open_dataset().objects[OBJECT]
    n = len(pairs["points"])
    perm = np.random.default_rng(0).permutation(n)
    nval = max(1, n // 10)
    sub = lambda idx: {k: v[idx] for k, v in pairs.items()}  # noqa: E731
    train_ds = CubePairDataset(sub(perm[nval:]) if n > 1 else pairs, obj, augment=True)
    val_ds = CubePairDataset(sub(perm[:nval]), obj, augment=False)

    bb = cfg.backbone
    neighbor_limits = calibrate_neighbors_stack_mode(train_ds, registration_collate_fn_stack_mode,
                                                     bb.num_stages, bb.init_voxel_size, bb.init_radius)

    def loader(ds, shuffle, workers):
        return build_dataloader_stack_mode(ds, registration_collate_fn_stack_mode, bb.num_stages,
                                           bb.init_voxel_size, bb.init_radius, neighbor_limits,
                                           batch_size=1, num_workers=workers, shuffle=shuffle)

    class Trainer(IterBasedTrainer):
        def __init__(self):
            super().__init__(cfg, max_iteration=iters, snapshot_steps=iters)
            self.register_loader(loader(train_ds, True, cfg.train.num_workers),
                                 loader(val_ds, False, cfg.test.num_workers))
            model = gt_model.create_model(cfg).cuda()
            sd = torch.load(init, map_location="cpu", weights_only=False)["model"]
            missing, unexpected = model.load_state_dict(sd, strict=False)
            self.logger.info(f"init weights {init} (missing {len(missing)}, unexpected {len(unexpected)})")
            self.register_model(model)
            opt = torch.optim.Adam(model.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)
            self.register_optimizer(opt)
            self.register_scheduler(build_warmup_cosine_lr_scheduler(
                opt, total_steps=iters, warmup_steps=cfg.optim.warmup_steps, eta_init=cfg.optim.eta_init,
                eta_min=cfg.optim.eta_min, grad_acc_steps=cfg.optim.grad_acc_steps))
            self.loss_func = gt_loss.OverallLoss(cfg).cuda()
            self.evaluator = gt_loss.Evaluator(cfg).cuda()

        def train_step(self, iteration, data_dict):
            output_dict = run_model(self.model, data_dict)
            loss_dict = self.loss_func(output_dict, data_dict)
            loss_dict.update(self.evaluator(output_dict, data_dict))
            return output_dict, loss_dict

        def val_step(self, iteration, data_dict):
            output_dict = run_model(self.model, data_dict)
            loss_dict = self.loss_func(output_dict, data_dict)
            loss_dict.update(self.evaluator(output_dict, data_dict))
            return output_dict, loss_dict

    t0 = time.time()
    trainer = Trainer()
    trainer.run()
    torch.save({"model": trainer.model.state_dict(), "iteration": iters, "setting": a.setting,
                "neighbor_limits": list(map(int, neighbor_limits)), "init": str(init),
                "n_train": len(train_ds), "domain": domain}, out)
    print(f"{a.setting}: {len(train_ds)} train / {len(val_ds)} val pairs, {iters} iters, "
          f"{time.time() - t0:.0f} s -> {out}")


if __name__ == "__main__":
    main()
