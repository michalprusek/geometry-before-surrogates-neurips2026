"""Appendix A: the settings of GeoWalk and GeoCMA on GuacaMol, read from the code that ran them.

``tab_hyper`` prints the constants of the two methods and the shared GuacaMol protocol. Every value is read from the
code of the released repository (``ctx.extra["repo_root"]``), never typed here:

  value                                      read from
  -----------------------------------------  ----------------------------------------------------------------------
  N0, T, N = N0 + T, subspace dimension      ``scripts/cr/run_cell.py`` RUN_DEFAULTS (rule R-1 of the runs)
  rho, alpha_max, alpha_min, radius rule     ``GeoWalkSolver.__init__`` keyword defaults (``src/spherevo/solver.py``)
  N1 = int(rho T), angle schedule, anchor    ``GeoWalkSolver.__init__`` / ``_angle`` / ``next_candidate`` source
  population, sigma0, mean, basis, lift      ``CMAESSubspaceSolver`` and ``_CMAESDeferredBase`` (solver.py), and the
                                             keywords that ``_create_solver`` passes (``scripts/run_benchmark.py``)
  CMA options                                the options dict of ``_CMAESDeferredBase.__init__``
  pycma version                              ``provenance.package_versions`` of the loaded GeoCMA result files
  tasks, seeds, per-run limit                ``scripts/cr/manifest.json``
  cold start, decoding cap, 30 s cap         ``generate_cold_start_ours`` / ``make_ours_bb`` / ``LatentBlackBox`` of
                                             run_benchmark.py; greedy decoding: ``SELFIESVAECodec.decode`` default

The factory call of each method is checked to pass nothing that would override a printed default (GeoWalk: only
``total_budget=n_iterations``; GeoCMA: ``subspace_dim``, ``population_size``, ``norm_mode``), and ``run_benchmark``
to pass ``n_iterations=max_iter``. A missing or changed construct raises ``HyperSourceError``: the table is never
printed from a guess. The settings on LassoBench and NAS-Bench-101 are those of their own runners and are given in
the text of the paper (Section 4.4), not here.
"""
from __future__ import annotations

import ast
import json
import os
import re

from . import numbers as N
from .context import Ctx
from .latex import WRAP, Table, wrapp

SOLVER = "src/spherevo/solver.py"
RUNNER = "scripts/run_benchmark.py"
RUN_CELL = "scripts/cr/run_cell.py"
MANIFEST = "scripts/cr/manifest.json"
CODEC = "src/codec/selfies_vae.py"


class HyperSourceError(RuntimeError):
    """A construct the table is read from is missing or changed."""


def _read(repo: str, rel: str) -> str:
    p = os.path.join(repo, rel)
    if not os.path.exists(p):
        raise HyperSourceError(f"{rel} not found under {repo}")
    with open(p) as f:
        return f.read()


def _class(tree: ast.Module, name: str) -> ast.ClassDef:
    for n in tree.body:
        if isinstance(n, ast.ClassDef) and n.name == name:
            return n
    raise HyperSourceError(f"class {name} not found")


def _method(cls: ast.ClassDef, name: str) -> ast.FunctionDef:
    for n in cls.body:
        if isinstance(n, ast.FunctionDef) and n.name == name:
            return n
    raise HyperSourceError(f"{cls.name}.{name} not found")


def _func(tree: ast.Module, name: str) -> ast.FunctionDef:
    for n in tree.body:
        if isinstance(n, ast.FunctionDef) and n.name == name:
            return n
    raise HyperSourceError(f"function {name} not found")


def init_defaults(tree: ast.Module, cls: str) -> dict:
    """Literal keyword defaults of ``cls.__init__`` (positional-or-keyword and keyword-only)."""
    f = _method(_class(tree, cls), "__init__")
    a = f.args
    out = {}
    pos = a.posonlyargs + a.args
    for arg, d in zip(pos[len(pos) - len(a.defaults):], a.defaults):
        out[arg.arg] = ast.literal_eval(d)
    for arg, d in zip(a.kwonlyargs, a.kw_defaults):
        if d is not None:
            out[arg.arg] = ast.literal_eval(d)
    return out


def factory_kwargs(tree: ast.Module, solver: str) -> dict:
    """{keyword: source} of the call that ``_create_solver`` returns under ``if solver_name == "<solver>":``."""
    f = _func(tree, "_create_solver")
    for node in ast.walk(f):
        if (isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
                and isinstance(node.test.left, ast.Name) and node.test.left.id == "solver_name"
                and len(node.test.comparators) == 1 and isinstance(node.test.comparators[0], ast.Constant)
                and node.test.comparators[0].value == solver):
            for st in node.body:
                if isinstance(st, ast.Return) and isinstance(st.value, ast.Call):
                    return {k.arg: ast.unparse(k.value) for k in st.value.keywords}
    raise HyperSourceError(f"_create_solver has no return for solver {solver!r}")


def call_keyword(tree: ast.Module, caller: str, callee: str, kw: str) -> str:
    """Source of keyword ``kw`` in the call of ``callee`` inside function ``caller``."""
    for node in ast.walk(_func(tree, caller)):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == callee:
            for k in node.keywords:
                if k.arg == kw:
                    return ast.unparse(k.value)
    raise HyperSourceError(f"{caller} does not call {callee}({kw}=...)")


def dict_call(tree: ast.Module, name: str) -> dict:
    """Literal keywords of a module-level ``NAME = dict(...)``."""
    for n in tree.body:
        if (isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
                and n.targets[0].id == name and isinstance(n.value, ast.Call)
                and isinstance(n.value.func, ast.Name) and n.value.func.id == "dict"):
            return {k.arg: ast.literal_eval(k.value) for k in n.value.keywords}
    raise HyperSourceError(f"{name} = dict(...) not found")


def _src(text: str, node) -> str:
    return ast.get_source_segment(text, node) or ""


def _need(text: str, pattern: str, what: str) -> re.Match:
    mo = re.search(pattern, text)
    if not mo:
        raise HyperSourceError(f"{what}: pattern {pattern!r} not found")
    return mo


def read_settings(repo: str) -> dict:
    """Every printed value, read from the code (see the module docstring)."""
    sol_txt, run_txt, cell_txt = _read(repo, SOLVER), _read(repo, RUNNER), _read(repo, RUN_CELL)
    sol, run, cell = ast.parse(sol_txt), ast.parse(run_txt), ast.parse(cell_txt)
    rd = dict_call(cell, "RUN_DEFAULTS")
    gw = init_defaults(sol, "GeoWalkSolver")
    cm = init_defaults(sol, "CMAESSubspaceSolver")
    base = init_defaults(sol, "_CMAESDeferredBase")
    # GeoWalk: the factory passes the budget and nothing else
    fk = factory_kwargs(run, "geowalk")
    extra = set(fk) - {"black_box", "x0", "y0", "total_budget"}
    if extra or fk.get("total_budget") != "n_iterations":
        raise HyperSourceError(f"the geowalk factory passes {fk}: a printed default may be overridden")
    if call_keyword(run, "run_benchmark", "_create_solver", "n_iterations") != "max_iter":
        raise HyperSourceError("run_benchmark does not pass n_iterations=max_iter")
    neutral = {"explore_steps": None, "direction_accumulation": False, "radius_override": None, "radius_scale": 1.0,
               "shell_eps": 0.0, "radius_range": None, "total_budget": None}
    bad = {k: gw.get(k) for k, d in neutral.items() if gw.get(k, "missing") != d}
    if bad:
        raise HyperSourceError(f"GeoWalkSolver defaults changed: {bad}")
    init = _src(sol_txt, _method(_class(sol, "GeoWalkSolver"), "__init__"))
    _need(init, r"self\.explore_steps = int\(explore_ratio \* total_budget\)", "GeoWalk exploration steps")
    ang = _src(sol_txt, _method(_class(sol, "GeoWalkSolver"), "_angle"))
    _need(ang, r"denom = max\(self\.total_budget - self\.explore_steps - 1, 1\)", "GeoWalk angle schedule")
    _need(ang, r"t = min\(refine_step / denom, 1\.0\)", "GeoWalk angle schedule")
    _need(ang, r"return self\.angle_init - \(self\.angle_init - self\.angle_final\) \* t", "GeoWalk angle schedule")
    nxt = _src(sol_txt, _method(_class(sol, "GeoWalkSolver"), "next_candidate"))
    _need(nxt, r"pts = np\.concatenate\(self\.history\[\"x\"\]", "GeoWalk anchor")
    _need(nxt, r"best_idx = int\(np\.argmax\(ys\)\)", "GeoWalk anchor")
    _need(init, r"else:  # \"mean\".*\n\s*self\._base_radius = self\.mean_norm \* radius_scale", "GeoWalk radius")
    base_init = _src(sol_txt, _method(_class(sol, "StepByStepSolver"), "__init__"))
    _need(base_init, r"self\.mean_norm = float\(np\.mean\(norms\)\)", "cold-start mean norm")
    # GeoCMA
    ck = factory_kwargs(run, "cma_es_subspace")
    if set(ck) - {"black_box", "x0", "y0", "subspace_dim", "population_size", "norm_mode"} or \
            ck.get("subspace_dim") != "subspace_dim" or ck.get("norm_mode") != "norm_mode":
        raise HyperSourceError(f"the cma_es_subspace factory passes {ck}")
    pop = ast.literal_eval(ck["population_size"])
    cinit = _src(sol_txt, _method(_class(sol, "CMAESSubspaceSolver"), "__init__"))
    _need(cinit, r"A, _ = np\.linalg\.qr\(np\.random\.randn\(self\.D, self\.d\)\)", "GeoCMA basis")
    _need(cinit, r"initial_mean = x0\[best_idx\] @ A", "GeoCMA initial mean")
    _need(cinit, r"initial_sigma=initial_sigma", "GeoCMA initial step size")
    lift = _src(sol_txt, _method(_class(sol, "CMAESSubspaceSolver"), "_cma_to_ambient"))
    for pat in (r"v_norm = _normalize\(v\)", r"u = v_norm @ self\._A\.T", r"u_norm = _normalize\(u\)",
                r"return \(u_norm \* ambient_norm\)", r"ambient_norm = self\.mean_norm"):
        _need(lift, pat, "GeoCMA lift")
    binit = _src(sol_txt, _method(_class(sol, "_CMAESDeferredBase"), "__init__"))
    opts = _need(binit, r"opts: dict = (\{[^}]*\})", "CMA options").group(1)
    _need(opts, r'"popsize": population_size', "CMA options")
    verbose = int(_need(opts, r'"verbose": (-?\d+)', "CMA options").group(1))
    _need(opts, r'"seed": cma_seed', "CMA options")
    _need(binit, r"cma_seed = int\(np\.random\.randint\(0, 2\*\*31\)\)", "CMA seed")
    _need(binit, r"_ = self\._cma\.ask\(\)\s*\n\s*self\._cma\.tell\(cs_proj\.tolist\(\), \(-cs_scores\)\.tolist\(\)\)",
          "GeoCMA pre-conditioning")
    nc = _src(sol_txt, _method(_class(sol, "_CMAESDeferredBase"), "next_candidate"))
    _need(nc, r"self\._cma\.tell\(self\._gen_raw, \[-s for s in self\._gen_scores\]\)", "GeoCMA deferred tell")
    # shared protocol
    man = json.loads(_read(repo, MANIFEST))
    arms = {a["arm"]: a for a in man["arms"]}
    for a in ("geowalk", "cma_es_subspace"):
        if a not in arms or arms[a]["solver"] != a or arms[a]["run_overrides"]:
            raise HyperSourceError(f"manifest arm {a} is not the plain solver {a}")
    timeouts = {arms[a]["timeout_s"] for a in ("geowalk", "cma_es_subspace")}
    if len(timeouts) != 1:
        raise HyperSourceError(f"the two methods have different per-run limits {timeouts}")
    cs = _src(run_txt, _func(run, "generate_cold_start_ours"))
    n_zinc = int(_need(cs, r"load_zinc_smiles\(n_samples=(\d+)\)", "cold-start pool").group(1))
    _need(cs, r"rng = np\.random\.RandomState\(seed\)", "cold-start draw")
    _need(cs, r"rng\.choice\(len\(zinc_smiles\), n_cold_start, replace=False\)", "cold-start draw")
    _need(cs, r"y0 = np\.array\(\[oracle_score_fn\(smi\) for smi in cs_smiles\]\)", "cold-start scores")
    cap = int(_need(run_txt, r"codec\.model\.max_string_length = (\d+)", "decoding cap").group(1))
    bb = _src(run_txt, _class(run, "LatentBlackBox"))
    alarm = int(_need(bb, r"signal\.alarm\((\d+)\)", "decode cap").group(1))
    # the alarm is set around the decoder call and cancelled in its ``finally`` before the oracle scores the decoded
    # molecules: decoding is time-limited, scoring is not (RUN_RULES 2026-10-29 (4); the table says so)
    _need(bb, r"signal\.alarm\(\d+\)\s*\n\s*try:\s*\n\s*smiles_list = self\.decode_fn\(z\)", "decode cap")
    fin = _need(bb, r"finally:\s*\n\s*signal\.alarm\(0\)", "decode cap cancelled").end()
    score_at = _need(bb, r"self\.score_fn\(", "oracle call").start()
    scoring_capped = not fin < score_at or bool(re.search(r"signal\.alarm\([1-9]", bb[fin:]))
    codec_txt = _read(repo, CODEC)
    codec = ast.parse(codec_txt)
    dec = init_defaults_method(codec, "SELFIESVAECodec", "decode")
    if dec.get("deterministic") is not True:
        raise HyperSourceError("SELFIESVAECodec.decode is not greedy by default")
    enc = _src(codec_txt, _method(_class(codec, "SELFIESVAECodec"), "encode"))
    _need(enc, r"mu, _sigma = self\.model\.encode\(tokens\)", "encoder mean")
    T, N0 = rd["max_iter"], rd["n_cold_start"]
    N1 = int(gw["explore_ratio"] * T)
    return {"N0": N0, "T": T, "N": N0 + T, "rho": gw["explore_ratio"], "N1": N1, "N2": T - N1,
            "intervals": max(T - N1 - 1, 1), "alpha_max": gw["angle_init"], "alpha_min": gw["angle_final"],
            "radius": gw["radius_estimator"], "subspace": rd["subspace_dim"], "cma_default_subspace": cm["subspace_dim"],
            "lambda": pop, "sigma0": cm["initial_sigma"], "norm_mode": rd["norm_mode"], "verbose": verbose,
            "base_sigma_default": base["initial_sigma"], "tasks": list(man["tasks"]), "seeds": list(man["seeds"]),
            "timeout_s": timeouts.pop(), "zinc_pool": n_zinc, "decode_cap": cap, "decode_seconds": alarm,
            "scoring_capped": scoring_capped}


def init_defaults_method(tree: ast.Module, cls: str, method: str) -> dict:
    """Literal keyword defaults of any method of a class."""
    f = _method(_class(tree, cls), method)
    a = f.args
    pos = a.posonlyargs + a.args
    out = {arg.arg: ast.literal_eval(d) for arg, d in zip(pos[len(pos) - len(a.defaults):], a.defaults)}
    out.update({arg.arg: ast.literal_eval(d) for arg, d in zip(a.kwonlyargs, a.kw_defaults) if d is not None})
    return out


def cma_version(ctx: Ctx) -> str:
    """pycma version(s) recorded in the loaded GeoCMA result files."""
    from .baselines import arm_envs
    if not ctx.has("cma_es_subspace"):
        return "not recorded"
    vs = sorted({dict(env[2]).get("cma") for env in arm_envs(ctx, "cma_es_subspace") if env} - {None})
    return " / ".join(vs) if vs else "not recorded"


def _seed_span(seeds) -> str:
    s = sorted(seeds)
    if s == list(range(s[0], s[-1] + 1)):
        return f"{s[0]}--{s[-1]} ({len(s)} seeds)"
    return ", ".join(map(str, s))


def tab_hyper(ctx: Ctx) -> Table | None:
    repo = ctx.extra.get("repo_root") or ""
    v = read_settings(repo)
    if v["norm_mode"] != "fixed":
        raise HyperSourceError(f"run default norm_mode {v['norm_mode']!r}: GeoCMA would not keep a fixed radius")
    # RUN_RULES 2026-10-29 (4): two columns; the caption says that every value is read from the released code
    t = Table("hyper", "tab:hyper", "", wrapp("4.0cm") + WRAP, ["Setting", "Value"], size="\\footnotesize",
              tabcolsep="4pt")
    a_max, a_min = f"{v['alpha_max']:.2f}", f"{v['alpha_min']:.2f}"
    t.group("\\geowalk{}")
    t.add(["Cold-start points $N_0$", str(v["N0"])])
    t.add(["Evaluations after the cold start $T$", str(v["T"])])
    t.add(["Budget $N=N_0+T$", str(v["N"])])
    t.add(["Exploration ratio $\\rho$", f"{v['rho']:.2f}"])
    t.add(["Exploration steps $N_1=\\lfloor\\rho T\\rfloor$",
           f"{v['N1']} uniform proposals on the shell"])
    t.add(["Refinement steps $T-N_1$", f"{v['N2']} geodesic steps"])
    t.add(["Angle $\\alpha_{\\max}\\to\\alpha_{\\min}$", f"${a_max}\\to{a_min}$ rad"])
    t.add(["Angle schedule", f"linear in the refinement step, over $\\max(T-N_1-1,1)={v['intervals']}$ intervals"])
    t.add(["Anchor of a refinement step", "best evaluated point so far, cold starts included"])
    t.add(["Radius $\\bar r$", f"{'mean' if v['radius'] == 'mean' else v['radius']} norm of the $N_0$ cold-start "
           "latents"])
    t.group("\\geocma{}")
    t.add(["Subspace dimension $\\tilde D$", str(v["subspace"])])
    t.add(["Basis $Q$", "QR decomposition of a Gaussian $D\\times\\tilde D$ matrix"])
    t.add(["Population size $\\lambda$", str(v["lambda"])])
    t.add(["Initial step size $\\sigma_0$", f"{v['sigma0']}"])
    t.add(["Initial mean", "$Q^\\top x$ of the best cold start"])
    t.add(["Pre-conditioning", f"the $N_0={v['N0']}$ cold starts, projected by $Q^\\top$, told as the first "
           "generation"])
    t.add(["Update", "deferred: one evaluation per step; a generation is told once $\\lambda$ evaluations have "
           "accumulated"])
    t.add(["Lift", "normalize in the subspace, map by $Q$, renormalize, scale to $\\bar r$"])
    t.add(["CMA-ES library and options", f"pycma {cma_version(ctx)}; population $\\lambda$, verbosity "
           f"${v['verbose']}$, seed drawn from NumPy's global stream (seeded by the run seed)"])
    t.group("Shared protocol (GuacaMol)")
    t.add(["Tasks", f"{len(v['tasks'])} GuacaMol tasks"])
    t.add(["Seeds", _seed_span(v["seeds"])])
    t.add(["Cold start", f"$N_0$ molecules drawn without replacement from the first {N.intfmt(v['zinc_pool'])} "
           "ZINC molecules (NumPy RandomState of the seed), encoded by the encoder mean, scored directly"])
    t.add(["Decoding", f"greedy, at most {v['decode_cap']} tokens"])
    t.add(["Time cap per proposal", f"decoding and scoring capped at {v['decode_seconds']} s" if v["scoring_capped"] else
           f"decoding capped at {v['decode_seconds']} s, scoring not time-limited"])
    t.add(["Time limit per run", f"{N.intfmt(v['timeout_s'])} s for both methods, a guard against hangs, not a "
           "setting of the methods"])
    t.caption = ("\\textbf{Settings of \\geowalk{} and \\geocma{} on GuacaMol.} These settings hold for every "
                 "task and seed. Every value is read from the released code that ran the two methods (the pycma "
                 "version from the result files), and a test checks the table against the code. The settings that differ on LassoBench and NAS-Bench-101 are given in "
                 "Section~\\ref{sec:structured}.")
    return t
