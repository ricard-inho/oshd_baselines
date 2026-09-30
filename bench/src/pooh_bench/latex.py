r"""Table bodies and pgfplots data blocks that paste into the paper template.

Missing values become ``\tbd``; ``S_sim`` on FT rows is ``--`` as in the template. The best
``S_real`` of each modality block is wrapped in ``\best{}``.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

__all__ = ["write_latex"]


def _f(v, nd: int) -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return r"\tbd"
    return f"{v:.{nd}f}"


def _table1(t1: pd.DataFrame) -> str:
    lines = []
    for mod, block in t1.groupby("modality", sort=False):
        best = block["S_real"].min()
        lines.append(r"\midrule")
        first_mod = True
        for meth, rows in block.groupby("method", sort=False):
            first_meth = True
            for _, r in rows.iterrows():
                c_mod = rf"\multirow{{{len(block)}}}{{*}}{{{mod}}}" if first_mod else ""
                c_meth = (rf"\multirow{{{len(rows)}}}{{*}}{{{meth}}}" if len(rows) > 1 else meth) if first_meth else ""
                s_real = _f(r["S_real"], 3)
                if math.isfinite(r["S_real"]) and r["S_real"] == best:
                    s_real = rf"\best{{{s_real}}}"
                s_sim = _f(r["S_sim"], 3) if r["setting"] == "ZS" else "--"
                cells = [c_mod, c_meth, rf"\{r['setting']}", _f(r["e_t_cm"], 2), _f(r["e_q_deg"], 2), s_sim,
                         s_real, _f(r["add_auc"], 3), _f(r["rate_hz"], 1)]
                lines.append(" & ".join(cells) + r" \\")
                first_mod = first_meth = False
    return "\n".join(lines[1:]) + "\n"  # the template already has \midrule after the header


def _table2(t2: pd.DataFrame) -> str:
    out = []
    for _, r in t2.iterrows():
        if r["key"] == "mocap":
            continue  # reference row is fixed in the template
        est = r["estimator"] or r"\tbd"
        out.append(f"{r['modality']} & {est} & {_f(r['mean_delta_a'], 3)} & {_f(r['p95_delta_a'], 3)} \\\\")
    return "\n".join(out) + "\n"


def _table3(t3: pd.DataFrame) -> str:
    out = []
    for meth, rows in t3.groupby("method", sort=False):
        for i, (_, r) in enumerate(rows.iterrows()):
            c = (rf"\multirow{{{len(rows)}}}{{*}}{{{meth}}}" if len(rows) > 1 else meth) if i == 0 else ""
            out.append(f"{c} & \\{r['setting']} & {_f(r['EPE'], 2)} & {_f(r['1PE'], 1)} & {_f(r['3PE'], 1)} "
                       f"& {_f(r['AE'], 2)} \\\\")
    return "\n".join(out) + "\n"


def _pgf(df: pd.DataFrame, cols: list[str]) -> str:
    """Whitespace table for ``\\pgfplotstableread{...}``; NaN -> ``nan`` (pgfplots skips it)."""
    lines = [" ".join(cols)]
    for _, r in df.iterrows():
        lines.append(" ".join("nan" if not math.isfinite(float(r[c])) else f"{float(r[c]):.6g}" for c in cols))
    return "\n".join(lines) + "\n"


def write_latex(out: Path, t1, t2, t3, fa, fb) -> dict[str, Path]:
    files = {
        "table1_pose.tex": _table1(t1),
        "table2_replay.tex": _table2(t2),
        "table3_flow.tex": _table3(t3),
        "fig1a_range.dat": _pgf(fa, ["x", "lo", "hi"] + [c for c in fa.columns if c not in ("x", "lo", "hi")]),
        "fig1b_angrate.dat": _pgf(fb, ["x", "lo", "hi"] + [c for c in fb.columns if c not in ("x", "lo", "hi")]),
    }
    paths = {}
    for name, text in files.items():
        p = out / name
        p.write_text(text)
        paths[name] = p
    return paths
