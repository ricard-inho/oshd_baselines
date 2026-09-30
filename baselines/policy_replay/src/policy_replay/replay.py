"""Run the frozen policy on reconstructed observations and write ``results/replay/<key>.parquet``."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .estimator import RigConstants, estimator_track, mocap_track
from .logs import PolicyLog, load_log
from .observation import build_observations, layout_names
from .tracks import PoseTrack, State, causal_state

__all__ = ["Policy", "default_config_path", "load_config", "replay_trajectory", "write_replay"]


def default_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "replay.yaml"


def load_config(path: str | Path | None = None) -> dict:
    return yaml.safe_load(Path(path or default_config_path()).read_text())


class Policy:
    """The exported rsl_rl actor (ONNX, CPU); returns the raw (unclipped) mean action."""

    def __init__(self, onnx_path: str | Path):
        import onnxruntime as ort

        self.sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        self.dim = int(self.sess.get_inputs()[0].shape[1])

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, np.float32)
        out = [self.sess.run(None, {self.inp: o[None]})[0][0] for o in obs]
        return np.asarray(out, np.float64).reshape(len(obs), -1)


def previous_actions(log: PolicyLog, lag: int) -> np.ndarray:
    prev = np.zeros_like(log.actions)
    if lag > 0:
        prev[lag:] = log.actions[:-lag]
    return prev


def replay_trajectory(policy: Policy, cfg: dict, log: PolicyLog, track: PoseTrack) -> tuple[np.ndarray, State, np.ndarray]:
    """(replayed clipped actions (K, 13), state, observations) at the logged policy steps."""
    tm = cfg["timing"]
    t_obs = log.t_ns - int(float(tm.get("obs_latency_ms", 0)) * 1e6)
    v = cfg["velocity"]
    s = causal_state(track, t_obs, float(v["window_ms"]), int(v.get("min_samples", 3)),
                     v.get("mode", "fit"), int(v.get("buffer_size", 30)))
    obs = build_observations(cfg, s, log.goal_xy, log.goal_heading, previous_actions(log, int(tm.get("prev_action_lag", 1))))
    if obs.shape[1] != policy.dim:
        raise ValueError(f"observation layout has {obs.shape[1]} entries, the policy expects {policy.dim}")
    act = np.full((len(obs), 13), np.nan)
    ok = s.valid & np.isfinite(obs).all(1)
    if ok.any():
        lo, hi = cfg["policy"].get("clip", [-1.0, 1.0])
        act[ok] = np.clip(policy(obs[ok]), lo, hi)
    return act, s, obs


def _frame(cfg: dict, name: str, log: PolicyLog, act: np.ndarray, s: State, ref: np.ndarray | None) -> pd.DataFrame:
    idx = list(cfg["actions"]["compare_idx"])
    names = cfg["actions"]["names"]
    d = act[:, idx] - log.actions[:, idx]
    df = pd.DataFrame({"trajectory": name, "t_ns": log.t_ns.astype(np.int64),
                       "delta_a": np.linalg.norm(d, axis=1), "has_estimate": s.valid,
                       "estimate_age_ms": s.age_ms})
    if ref is not None:
        df["delta_a_vs_mocap_replay"] = np.linalg.norm(act[:, idx] - ref[:, idx], axis=1)
    for j, i in enumerate(idx):
        df[f"d_{names[i]}"] = d[:, j]
    for i, n in enumerate(names):
        df[f"a_replay_{n}"] = act[:, i]
        df[f"a_logged_{n}"] = log.actions[:, i]
    df.loc[~s.valid, "delta_a"] = np.nan
    return df


def write_replay(results: Path, key: str, ds, names: list[str], cfg: dict, policy: Policy,
                 source: str, label: str, preds: pd.DataFrame | None = None, latency: bool = False,
                 obj=None, sym_T=None) -> tuple[Path, dict]:
    """Replay every trajectory in ``names``; ``preds`` None = the privileged mocap row."""
    rc = RigConstants(ds, cfg)
    frames, per_traj = [], {}
    for name in names:
        traj = ds[name]
        log = load_log(traj, cfg)
        mtrack, msrc = mocap_track(traj, cfg, rc)
        ref_act, ref_s, _ = replay_trajectory(policy, cfg, log, mtrack)
        if preds is None:
            act, s, ref = ref_act, ref_s, None
            info = {"pose_source": msrc}
        else:
            track, info = estimator_track(traj, preds, cfg, rc, obj, sym_T, latency)
            act, s, _ = replay_trajectory(policy, cfg, log, track)
            ref = ref_act
        df = _frame(cfg, name, log, act, s, ref)
        info.update({"n_steps": len(df), "n_steps_without_estimate": int((~s.valid).sum()),
                     "mean_delta_a": float(np.nanmean(df["delta_a"])) if df["delta_a"].notna().any() else None,
                     "median_estimate_age_ms": float(np.nanmedian(s.age_ms)) if s.valid.any() else None})
        per_traj[name] = info
        frames.append(df)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["trajectory", "t_ns", "delta_a"])
    path = Path(results) / "replay" / f"{key}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    d = out["delta_a"].to_numpy(float)
    d = d[np.isfinite(d)]
    meta = {
        "estimator": label, "source": source, "latency_runtime_ms": bool(latency),
        "policy": str(cfg["policy"]["onnx"]), "compare_idx": cfg["actions"]["compare_idx"],
        "observation_layout": layout_names(cfg),
        "mean_delta_a": float(d.mean()) if len(d) else None,
        "p95_delta_a": float(np.percentile(d, 95)) if len(d) else None,
        "n_steps": len(out), "n_steps_without_estimate": int((~out["has_estimate"]).sum()) if len(out) else 0,
        "trajectories": per_traj, "config": cfg,
    }
    if "delta_a_vs_mocap_replay" in out:
        r = out["delta_a_vs_mocap_replay"].to_numpy(float)
        r = r[np.isfinite(r)]
        meta["mean_delta_a_vs_mocap_replay"] = float(r.mean()) if len(r) else None
        meta["p95_delta_a_vs_mocap_replay"] = float(np.percentile(r, 95)) if len(r) else None
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2, default=str))
    return path, meta
