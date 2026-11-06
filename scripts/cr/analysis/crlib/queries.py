"""A small query language over the loaded arms.

One query describes how a single earlier number was computed (arm(s), cell mask, cell filter,
level convention, statistic). The same query is evaluated
  * in --legacy mode on the main namespace (the old stored files) -> legacy_check,
  * in camera-ready mode on the 'posted:' namespace (the posted generation loaded from
    sources_legacy.json) -> the 'recomputed from the stored posted files' column of
    tab_posted_vs_printed.
Query types: level, level_agg, paired, interaction, n, bh, paired_agg, number (the run's own registry; with
"field": "text" the printed text instead of the value), json.
Mask: {"kind": "own" | "pair" | "joint", "arms": [...], "filter": "clean" | "completed",
       "task": "osmb", "seeds": "42-51"}.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

from . import stats
from .config import parse_seeds


class QueryError(Exception):
    pass


def _arm(name: str, ns: str) -> str:
    return f"{ns}:{name}" if ns else name


def _mask(ctx, spec: dict, default_arms: list[str], ns: str) -> set:
    spec = spec or {}
    kind = spec.get("kind", "pair")
    if kind == "own":
        arms = default_arms[:1]
    elif kind == "pair":
        arms = default_arms
    elif kind == "joint":
        arms = [_arm(a, ns) for a in spec["arms"]]
    else:
        raise QueryError(f"mask kind {kind}")
    for a in arms:
        if not ctx.has(a):
            raise QueryError(f"arm {a} absent")
    cells = ctx.eng.joint(arms, spec.get("filter"))
    if "task" in spec:
        cells = {c for c in cells if c[0] == spec["task"]}
    if "seeds" in spec:
        ss = set(parse_seeds(spec["seeds"]))
        cells = {c for c in cells if c[1] in ss}
    return cells


# 'wins' / 'losses' / 'p' reproduce EARLIER numbers, which all used exact float differences, so they map to
# the exact-difference counts and the exact-difference Wilcoxon p; the camera-ready rule (|diff| <= 1e-6 is
# a tie, in W/L and in the cell-level test) is 'wins_tol' / 'losses_tol' / 'p_tol'.
_PSTAT = {"delta_flat": "delta_flat", "delta_taskmean": "delta_taskmean", "d": "d", "p": "p_exact",
          "p_tol": "p",
          "wins": "wins_exact", "losses": "losses_exact", "wins_tol": "wins", "losses_tol": "losses",
          "n": "n", "tasks_won": "tasks_won", "tasks_lost": "tasks_lost", "p_task": "p_task",
          "level_a": "level_a", "level_b": "level_b", "n_tasks": "n_tasks"}


def evaluate(ctx, q: dict, ns: str = "", registries: dict | None = None):
    """Returns (value, n) for the query."""
    t = q["type"]
    conv = q.get("conv", "taskmean")
    N = int(q.get("N", 30))
    if t == "level":
        arm = _arm(q["arm"], ns)
        cells = _mask(ctx, q.get("mask", {"kind": "own"}), [arm], ns)
        return stats.level(ctx.eng.vals(arm, cells, N), cells, conv), len(cells)
    if t == "paired":
        a, b = _arm(q["a"], ns), _arm(q["b"], ns)
        cells = _mask(ctx, q.get("mask", {"kind": "pair"}), [a, b], ns)
        pr = stats.paired(ctx.eng.vals(a, cells, N), ctx.eng.vals(b, cells, N), cells, conv)
        return getattr(pr, _PSTAT[q["stat"]]), pr.n
    if t == "interaction":
        arms = {k: _arm(q[k], ns) for k in ("SA", "SC", "BA", "BC")}
        cells = _mask(ctx, q.get("mask", {"kind": "joint", "arms": [q[k] for k in ("SA", "SC", "BA", "BC")]}),
                      list(arms.values()), ns)
        v = {k: ctx.eng.vals(a, cells, N) for k, a in arms.items()}
        inter = {c: (v["SA"][c] - v["BA"][c]) - (v["SC"][c] - v["BC"][c]) for c in cells}
        pr = stats.paired(inter, {c: 0.0 for c in cells}, cells, conv)
        return getattr(pr, _PSTAT[q["stat"]]), pr.n
    if t == "n":
        cells = _mask(ctx, q["mask"], [], ns)
        return len(cells), len(cells)
    if t == "bh":
        ps = []
        for a, b in q["family"]:
            a, b = _arm(a, ns), _arm(b, ns)
            cells = _mask(ctx, q.get("mask", {"kind": "pair"}), [a, b], ns)
            ps.append(stats.paired(ctx.eng.vals(a, cells), ctx.eng.vals(b, cells), cells, conv).p_exact)
        return stats.bh(ps)[q["target"]], len(ps)
    if t == "paired_agg":
        vals = []
        for a in q["a_list"]:
            a2, b2 = _arm(a, ns), _arm(q["b"], ns)
            cells = _mask(ctx, q.get("mask", {"kind": "pair"}), [a2, b2], ns)
            pr = stats.paired(ctx.eng.vals(a2, cells, N), ctx.eng.vals(b2, cells, N), cells, conv)
            vals.append(getattr(pr, _PSTAT[q["stat"]]))
        return {"max": max, "min": min}[q["agg"]](vals), len(vals)
    if t == "level_agg":
        vals = []
        for a in q["a_list"]:
            arm = _arm(a, ns)
            cells = _mask(ctx, q.get("mask", {"kind": "own"}), [arm], ns)
            vals.append(stats.level(ctx.eng.vals(arm, cells, N), cells, conv))
        return {"max": max, "min": min}[q["agg"]](vals), len(vals)
    if t == "number":
        if ns:
            raise QueryError("number queries read the run's own registry; not evaluable on a namespace")
        reg = (registries or {}).get(q.get("variant", "default"))
        if reg is None or q["name"] not in reg.records:
            raise QueryError(f"number {q['name']} not in registry {q.get('variant', 'default')}")
        r = reg.records[q["name"]]
        if q.get("field") == "text":          # the printed text of the number (digit checks; cmp 'exact')
            return r["text"], r["n"]
        return r["value"], r["n"]
    if t == "json":
        root = ctx.extra.get("inputs_root", "")
        try:
            d = json.load(open(os.path.join(root, q["file"])))
        except OSError as e:
            raise QueryError(f"stored input {q['file']} is not readable under {root!r}: {e}") from e
        for k in q["path"]:
            d = d[k]
        return d, None
    raise QueryError(f"unknown query type {t}")


def _decimals(printed: str) -> int:
    s = printed.strip().lstrip("+-$").replace("\\,", "")
    if "e" in s.lower():
        return -1
    return len(s.split(".")[1]) if "." in s else 0


def compare(entry: dict, value) -> tuple[str, str]:
    """-> (status, detail). status in PASS, FAIL, KNOWN-DIFF, NA."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "NA", "no value"
    exp = entry.get("value")
    cmp = entry.get("cmp", "abs")
    printed = str(entry.get("printed", ""))
    if cmp == "abs":
        tol = entry.get("tol")
        if tol is None:
            dec = _decimals(printed)
            tol = 0.5 * 10 ** (-dec) if dec >= 0 else abs(exp) * 0.05
        ok = abs(float(value) - float(exp)) <= tol + 1e-12
        detail = f"|{float(value):.6g} - {exp}| <= {tol:g}"
    elif cmp == "absval":
        tol = entry.get("tol") or 0.5 * 10 ** (-_decimals(printed))
        ok = abs(abs(float(value)) - float(exp)) <= tol + 1e-12
        detail = f"||{float(value):.6g}| - {exp}| <= {tol:g}"
    elif cmp == "exact":
        ok = value == exp
        detail = f"{value} == {exp}"
    elif cmp == "lt":
        ok = float(value) < float(exp)
        detail = f"{float(value):.3g} < {exp}"
    elif cmp == "sig2":
        ok = f"{float(value):.1e}" == f"{float(exp):.1e}"
        detail = f"{float(value):.2e} ~ {exp} (2 significant digits)"
    else:
        raise QueryError(cmp)
    if ok:
        return "PASS", detail
    return ("KNOWN-DIFF" if entry.get("known_diff") else "FAIL"), detail


def fmt_value(v) -> str:
    if v is None:
        return "--"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, str):
        return v
    v = float(v)
    if math.isnan(v):
        return "--"
    if v != 0 and abs(v) < 1e-3:
        return f"{v:.2e}"
    return f"{v:.4f}"
