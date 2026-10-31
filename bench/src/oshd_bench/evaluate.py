"""Turn prediction files into one CSV + JSON per paper table / figure panel.

Outputs (``results/tables/``):

* ``table1_pose.csv``   one row per (modality, method, setting): e_t_cm, e_q_deg, S_sim,
  S_real, add_auc, rate_hz, coverage, n_samples.
* ``fig1a_range.csv``   S_real per range bin, one column per group (+ ``<group>_glare``).
* ``fig1b_angrate.csv`` S_real per relative angular-rate bin, one column per group.
* ``table2_replay.csv`` mean / p95 ||delta a|| per modality.
* ``table3_flow.csv``   EPE, 1PE, 3PE, AE per (method, setting).
* ``*.tex``             the table bodies with the same row order as the paper template.

Missing inputs leave NaN cells (written as ``nan``, rendered as ``\\tbd`` in the .tex).
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from oshd import OshDataset

from .gt import gt_poses
from .metrics import FlowStats, auc, pose_errors, speed_score
from .predictions import load_predictions, prediction_path
from .symmetry import symmetry_for

__all__ = ["load_config", "gt_path", "open_dataset", "repo_root", "results_dir", "Evaluator"]

KEY = ["trajectory", "t_ns", "object"]


def load_config(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text())


def repo_root() -> Path:
    """The oshd_baselines checkout (the directory holding configs/bench.yaml)."""
    for p in [Path.cwd(), *Path.cwd().parents, *Path(__file__).resolve().parents]:
        if (p / "configs" / "bench.yaml").exists():
            return p
    return Path.cwd()


def open_dataset(cfg: dict | None = None) -> OshDataset:
    """The benchmark dataset: ``$OSH_DATASET_ROOT``, else ``dataset.root`` of the config. Its
    splits come from the frozen benchmark ``dataset.benchmark`` (``$OSHD_BENCHMARK``; "none" =
    the dataset's ``splits.json``)."""
    cfg = cfg if cfg is not None else load_config(repo_root() / "configs" / "bench.yaml")
    d = cfg.get("dataset", {})
    bench = os.environ.get("OSHD_BENCHMARK", d.get("benchmark"))
    return OshDataset(os.environ.get("OSH_DATASET_ROOT") or d.get("root"),
                       repo_id=d.get("repo_id", "r3m3c3/oshd"),
                       benchmark=None if bench in (None, "", "none") else str(bench))


def results_dir(cfg: dict | None = None) -> Path:
    """``$OSHD_BENCH_RESULTS``, else ``<repo>/<results_dir>``."""
    cfg = cfg if cfg is not None else load_config(repo_root() / "configs" / "bench.yaml")
    return Path(os.environ.get("OSHD_BENCH_RESULTS") or repo_root() / cfg.get("results_dir", "results"))


def gt_path(results: str | Path, sensor: str, domain: str = "real", split: str = "test") -> Path:
    return Path(results) / "gt" / sensor / f"{domain}_{split}.parquet"


def _nan_to_none(o):
    if isinstance(o, float) and not math.isfinite(o):
        return None
    if isinstance(o, dict):
        return {k: _nan_to_none(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_nan_to_none(v) for v in o]
    return o


class Evaluator:
    def __init__(self, cfg: dict, root: str | Path = ".", dataset: OshDataset | None = None):
        self.cfg = cfg
        self.root = Path(root)
        # OSHD_BENCH_RESULTS / OSH_DATASET_ROOT override the config (used by the dev dataset)
        self.results = Path(os.environ.get("OSHD_BENCH_RESULTS") or self.root / cfg.get("results_dir", "results"))
        self.out = self.results / "tables"
        self._ds = dataset
        self._sym: dict = {}
        self._pts: dict = {}
        self.per_sample: dict[tuple, pd.DataFrame] = {}

    # ------------------------------------------------------------- helpers
    @property
    def ds(self) -> OshDataset:
        if self._ds is None:
            self._ds = open_dataset(self.cfg)
        return self._ds

    def _object(self, name: str):
        if name not in self._sym:
            obj = self.ds.objects[name]
            self._sym[name] = symmetry_for(obj, self.cfg.get("symmetries", {}).get(name))
            n = int(self.cfg.get("metrics", {}).get("add_model_points", 1000))
            rng = np.random.default_rng(0)
            v = obj.vertices
            self._pts[name] = v[rng.choice(len(v), size=min(n, len(v)), replace=False)]
        return self._sym[name], self._pts[name]

    def score_samples(self, gt: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
        """GT rows joined with predictions + per-sample errors (NaN where no prediction)."""
        m = self.cfg.get("metrics", {})
        pred = pred.sort_values("score", ascending=False).drop_duplicates(KEY)
        df = gt.merge(pred[KEY + ["T", "runtime_ms"]].rename(columns={"T": "T_pred"}), on=KEY, how="left")
        for c in ("e_t", "e_q", "add", "speed"):
            df[c] = np.nan
        has = df["T_pred"].notna().to_numpy()
        for name, idx in df[has].groupby("object").groups.items():
            sym, pts = self._object(name)
            sub = df.loc[idx]
            Tp = np.stack([np.asarray(t, float).reshape(4, 4) for t in sub["T_pred"]])
            Tg = gt_poses(sub)
            for s in range(0, len(sub), 256):  # chunk: (N, 24, P, 3) intermediates
                err = pose_errors(Tp[s:s + 256], Tg[s:s + 256], sym, pts)
                rows = sub.index[s:s + 256]
                df.loc[rows, "e_t"] = err.e_t
                df.loc[rows, "e_q"] = err.e_q
                df.loc[rows, "add"] = err.add
                df.loc[rows, "speed"] = speed_score(err, float(m.get("speed_t_threshold", 2.173e-3)),
                                                    float(m.get("speed_q_threshold_deg", 0.169)))
        return df

    def summarize(self, df: pd.DataFrame) -> dict:
        m = self.cfg.get("metrics", {})
        ok = df["e_t"].notna()
        rt = df.loc[ok, "runtime_ms"].dropna()
        return {
            "e_t_cm": float(df.loc[ok, "e_t"].mean() * 100) if ok.any() else float("nan"),
            "e_q_deg": float(df.loc[ok, "e_q"].mean()) if ok.any() else float("nan"),
            # medians: the paper reports means, but a few gross failures dominate them
            "e_t_cm_median": float(df.loc[ok, "e_t"].median() * 100) if ok.any() else float("nan"),
            "e_q_deg_median": float(df.loc[ok, "e_q"].median()) if ok.any() else float("nan"),
            "S": float(df.loc[ok, "speed"].mean()) if ok.any() else float("nan"),
            "add_auc": auc(df.loc[ok, "add"].to_numpy(), len(df), float(m.get("add_auc_max_m", 0.10)))
            if len(df) else float("nan"),
            "rate_hz": float(1000.0 / rt.median()) if len(rt) and rt.median() > 0 else float("nan"),
            "coverage": float(ok.mean()) if len(df) else float("nan"),
            "n_samples": len(df),
        }

    def _eval(self, sensor, modality, method, setting, domain) -> dict | None:
        g = gt_path(self.results, sensor, domain)
        p = prediction_path(self.results, modality, method, setting, domain)
        if not (g.exists() and p.exists()):
            return None
        df = self.score_samples(pd.read_parquet(g), load_predictions(p))
        self.per_sample[(modality, method, setting, domain)] = df
        return self.summarize(df)

    # -------------------------------------------------------------- tables
    def table1(self) -> pd.DataFrame:
        rows = []
        for block in self.cfg["table1"]:
            for meth in block["methods"]:
                for setting in meth["settings"]:
                    real = self._eval(block["sensor"], block["modality"], meth["key"], setting, "real")
                    sim = self._eval(block["sensor"], block["modality"], meth["key"], setting, "synthetic") \
                        if setting == "ZS" else None
                    nan = float("nan")
                    r = real or {}
                    rows.append({
                        "modality": block["label"], "modality_key": block["modality"], "group": block["group"],
                        "method": meth["label"], "method_key": meth["key"], "setting": setting,
                        "e_t_cm": r.get("e_t_cm", nan), "e_q_deg": r.get("e_q_deg", nan),
                        "e_t_cm_median": r.get("e_t_cm_median", nan), "e_q_deg_median": r.get("e_q_deg_median", nan),
                        "S_sim": (sim or {}).get("S", nan), "S_real": r.get("S", nan),
                        "gap": ((r.get("S", nan) - sim["S"]) if sim else nan),
                        "add_auc": r.get("add_auc", nan), "rate_hz": r.get("rate_hz", nan),
                        "coverage": r.get("coverage", nan), "n_samples": r.get("n_samples", 0),
                    })
        return pd.DataFrame(rows)

    def _best_per_group(self, t1: pd.DataFrame) -> dict[str, tuple]:
        pins = self.cfg.get("fig1", {}).get("pin", {}) or {}
        best = {}
        for group in self.cfg.get("fig1", {}).get("groups", []):
            if group in pins:
                mod, meth, setting = pins[group].split("/")
                best[group] = (mod, meth, setting)
                continue
            cand = t1[(t1["group"] == group) & t1["S_real"].notna()]
            if len(cand):
                r = cand.loc[cand["S_real"].idxmin()]
                best[group] = (r["modality_key"], r["method_key"], r["setting"])
        return best

    def fig1(self, t1: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
        f = self.cfg.get("fig1", {})
        best = self._best_per_group(t1)
        min_n = int(f.get("min_samples_per_bin", 5))
        glare = f.get("glare_label", "glare")

        def binned(df, col, edges):
            out = np.full(len(edges) - 1, np.nan)
            if df is None or not len(df):
                return out
            b = np.digitize(df[col].to_numpy(), edges) - 1
            for i in range(len(edges) - 1):
                s = df["speed"].to_numpy()[(b == i)]
                s = s[np.isfinite(s)]
                if len(s) >= min_n:
                    out[i] = s.mean()
            return out

        def table(col, edges, split_glare):
            edges = np.asarray(edges, float)
            mid = np.sqrt(edges[:-1] * edges[1:]) if col == "ang_rate_dps" else (edges[:-1] + edges[1:]) / 2
            t = pd.DataFrame({"lo": edges[:-1], "hi": edges[1:], "x": mid})
            for group in f.get("groups", []):
                df = self.per_sample.get((*best[group], "real")) if group in best else None
                if split_glare:
                    nom = None if df is None else df[df["illumination"] != glare]
                    gl = None if df is None else df[df["illumination"] == glare]
                    t[group] = binned(nom, col, edges)
                    t[f"{group}_glare"] = binned(gl, col, edges)
                else:
                    t[group] = binned(df, col, edges)
            return t

        a = table("range_m", f.get("range_bins_m", [0, 1, 2, 3]), True)
        b = table("ang_rate_dps", f.get("ang_rate_bins_dps", [0.1, 1, 10]), False)
        return a, b, {g: "/".join(v) for g, v in best.items()}

    def table2(self) -> pd.DataFrame:
        """Δa = ‖a(estimator) − a(policy replayed on mocap)‖, so the mocap row is 0 by construction
        and observation-rebuild error cancels. ``*_vs_logged`` (‖a − a_logged‖) is kept as the
        rebuild check; on the mocap row it is the number to report next to the table."""

        def stats(df: pd.DataFrame, col: str) -> tuple[float, float, int]:
            d = df[col].to_numpy(float) if col in df else np.array([])
            d = d[np.isfinite(d)]
            return (float(d.mean()), float(np.percentile(d, 95)), len(d)) if len(d) else (float("nan"),) * 2 + (0,)

        rows = []
        for r in self.cfg.get("table2", {}).get("rows", []):
            p = self.results / "replay" / f"{r['key']}.parquet"
            est, (mean, p95, n), (lmean, lp95) = "", (float("nan"), float("nan"), 0), (float("nan"), float("nan"))
            if p.exists():
                df = pd.read_parquet(p)
                lmean, lp95, _ = stats(df, "delta_a")
                if r["key"] == "mocap":
                    mean, p95, n = 0.0, 0.0, int(df["delta_a"].notna().sum())
                else:
                    mean, p95, n = stats(df, "delta_a_vs_mocap_replay")
                meta = p.with_suffix(".json")
                if meta.exists():
                    est = json.loads(meta.read_text()).get("estimator", "")
            rows.append({"modality": r["label"], "key": r["key"], "estimator": est,
                         "mean_delta_a": mean, "p95_delta_a": p95, "n_steps": n,
                         "mean_delta_a_vs_logged": lmean, "p95_delta_a_vs_logged": lp95})
        return pd.DataFrame(rows)

    def table3(self) -> pd.DataFrame:
        rows = []
        for m in self.cfg.get("table3", []):
            for setting in m["settings"]:
                p = self.results / "flow" / m["key"] / setting / "real_test.parquet"
                s = {"EPE": float("nan"), "1PE": float("nan"), "3PE": float("nan"), "AE": float("nan"),
                     "n_pixels": 0}
                if p.exists():
                    df = pd.read_parquet(p)
                    tot = FlowStats(int(df["n"].sum()), float(df["sum_epe"].sum()), int(df["n_gt1"].sum()),
                                    int(df["n_gt3"].sum()), float(df["sum_ae_deg"].sum()))
                    s = tot.summary()
                rows.append({"method": m["label"], "method_key": m["key"], "setting": setting, **s})
        return pd.DataFrame(rows)

    # ---------------------------------------------------------------- run
    def run(self) -> dict[str, Path]:
        self.out.mkdir(parents=True, exist_ok=True)
        t1 = self.table1()
        fa, fb, best = self.fig1(t1)
        t2 = self.table2()
        t3 = self.table3()
        files = {}
        for name, df in [("table1_pose", t1), ("fig1a_range", fa), ("fig1b_angrate", fb),
                         ("table2_replay", t2), ("table3_flow", t3)]:
            p = self.out / f"{name}.csv"
            df.to_csv(p, index=False, float_format="%.6g")
            (self.out / f"{name}.json").write_text(json.dumps(_nan_to_none(df.to_dict("records")), indent=2))
            files[name] = p
        (self.out / "fig1_best.json").write_text(json.dumps(best, indent=2))
        from .latex import write_latex

        files.update(write_latex(self.out, t1, t2, t3, fa, fb))
        return files
