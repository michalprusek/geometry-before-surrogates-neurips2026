"""Sources configuration.

A sources file (JSON, committed next to make_tables.py) maps every arm to exactly one
directory and one anchored filename pattern. Nothing is globbed by solver name, so the
loader can never pick a file from another generation, a budget run (``_b1000``), a
subspace ablation (``_sd8``) or another codec that happens to share a solver field.

Schema (``"schema": "cr-sources-v1"``)::

    {
      "name": "...", "description": "...",
      "roots": {"CR": "/abs/path", "NFS": "...", "INPUTS": "..."},   # ${NAME} substitution
      "tasks": ["adip", ...],              # short task ids (default: the 11 paper tasks)
      "seeds": "42-81",                    # or a list
      "cell_filter": "completed" | "clean",
      "include": [{"namespace": "posted", "file": "sources_legacy.json"}],   # optional
      "arms": {
        "<arm>": {
          "label": "GeoWalk",              # human label (tables use their own labels)
          "kind": "cr" | "stored",         # cr arms carry the tagged commit (provenance require)
          "format": "ours" | "external",
          "dir": "${CR}/geowalk",
          "pattern": "ours_geowalk_{task}_s{seed}_128d.json",
          "task_key": "poli" | "short",    # which task token the file name uses
          "solver_field": "geowalk" | null,  # expected JSON 'solver'/'method' value (null: not checked)
          "status_rule": "require_field" | "absent_is_completed",
          "provenance": "require" | "record",
          "seeds": "42-51",                # optional per-arm seed range. It is NOT intersected with the
                                           # config-level default (the replication arms '<arm>__s82' run
                                           # seeds 82-121); only a command-line --seeds subset narrows it
          "tasks": ["adip", ...],          # optional per-arm override
          "known_missing": ["siga:79"],    # cells declared missing (anything else missing fails)
          "required": true,                # false: the arm may be absent (reported loudly)
          "trace_len": 30,                 # 1000 for the N = 1000 budget runs
          "budget": 30,                    # expected 'budget' field for the clean filter
          "coldstart_from": "geowalk",     # external format: arm whose best_trace[:5] is the
                                           # shared cold-start prefix of the trajectory
          "cr_arm": "geowalk",             # camera-ready arms: provenance.cr_arm must equal this
                                           # (files are grouped by arm id, not by solver)
          "commit_root": "CRCTL",          # which root's expected commit the files must carry; default =
                                           # the root named first in "dir" (${CR}, ${CRCPU}, ${CRCTL}). The
                                           # main campaign and the control block were run from two tagged
                                           # commits; each root has its own expected commit
                                           # (make_tables.py --expected-commit ROOT=SHA)
          "fallback_rule": "recipe_step_fallback",   # how a cell's fallback steps are counted (crlib/loader.py):
                                           # "surrogate_fallback_count" (default: the file's field) or
                                           # "recipe_step_fallback" (steps of published_recipe.steps whose
                                           # 'fallback' is set; the rerun arms of 2026-11-02, whose
                                           # surrogate_fallback_count also counts invalid molecules)
          "expect": {"subspace_dim": 8,    # fields that must hold these values (fatal otherwise)
                     "provenance.run_overrides": {"subspace_dim": 8}},
          "display": {"label": "...",      # how tables must name the files this arm points at,
                      "support": "..."},   # when they are NOT the arm's own method (legacy
                                           # substitutes, e.g. the July RAASP files)
          "note": "..."
        }
      }
    }
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from dataclasses import dataclass, field

from . import TASKS_T11

SCHEMA = "cr-sources-v1"
#: how a cell's fallback steps are counted: the file's surrogate_fallback_count (default), or the steps of
#: published_recipe.steps whose 'fallback' entry is set (the rerun arms of 2026-11-02 count invalid molecules in
#: surrogate_fallback_count, which the other arms do not; crlib/loader.py)
FALLBACK_RULES = ("surrogate_fallback_count", "recipe_step_fallback")
# Every GuacaMol task token that can appear in a result-file name (11 paper tasks + the
# E-1 / budget extras). Used only to classify excluded files in the loader report.
EXTRA_POLI = [
    "valsartan_smarts", "albuterol_similarity", "celecoxib_rediscovery", "isomers_c7h8n2o2",
    "isomers_c9h10n2o2pf2cl", "logp", "mestranol_similarity", "qed", "thiothixene_rediscovery",
    "troglitazone_rediscovery",
]


class ConfigError(Exception):
    pass


def parse_seeds(spec) -> list[int]:
    if spec is None:
        return []
    if isinstance(spec, str):
        out: list[int] = []
        for part in spec.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                a, b = part.split("-")
                out.extend(range(int(a), int(b) + 1))
            else:
                out.append(int(part))
        return out
    return [int(s) for s in spec]


def _subst(s: str, roots: dict) -> str:
    def rep(m):
        k = m.group(1)
        if k not in roots:
            raise ConfigError(f"unknown root ${{{k}}} in {s!r}")
        return roots[k]
    return re.sub(r"\$\{([A-Z0-9_]+)\}", rep, s)


@dataclass
class ArmSpec:
    name: str
    label: str
    kind: str
    fmt: str
    dir: str
    pattern: str
    task_key: str
    solver_field: str | None
    status_rule: str
    provenance: str
    seeds: list[int]
    tasks: list[str]
    known_missing: set
    required: bool
    trace_len: int
    budget: int | None
    coldstart_from: str | None
    note: str
    namespace: str = ""
    regex: re.Pattern = field(default=None, repr=False)
    prefix: str = ""
    cr_arm: str | None = None
    expect: dict = field(default_factory=dict)
    display: dict = field(default_factory=dict)
    commit_root: str | None = None      # root whose expected commit applies (camera-ready arms)
    declared_seeds: bool = False        # the arm has its own "seeds" entry
    fallback_rule: str = "surrogate_fallback_count"   # how fallback steps are counted (FALLBACK_RULES)

    def build_regex(self) -> None:
        if self.pattern.count("{task}") != 1 or self.pattern.count("{seed}") != 1:
            raise ConfigError(f"{self.name}: pattern must contain {{task}} and {{seed}} once: {self.pattern}")
        tokens = list(TASKS_T11.values()) if self.task_key == "poli" else list(TASKS_T11.keys())
        tokens = sorted(tokens, key=len, reverse=True)
        out = "^"
        rest = self.pattern
        self.prefix = rest.split("{", 1)[0]
        for piece in re.split(r"(\{task\}|\{seed\}|\{dim\})", rest):
            if piece == "{task}":
                out += "(?P<task>" + "|".join(re.escape(t) for t in tokens) + ")"
            elif piece == "{seed}":
                out += r"(?P<seed>\d+)"
            elif piece == "{dim}":
                out += r"(?P<dim>\d+)"
            else:
                out += re.escape(piece)
        out += "$"
        self.regex = re.compile(out)

    def short_task(self, token: str) -> str:
        if self.task_key == "poli":
            for k, v in TASKS_T11.items():
                if v == token:
                    return k
            raise KeyError(token)
        return token

    def expected_cells(self) -> set:
        return {(t, s) for t in self.tasks for s in self.seeds}

    def as_dict(self) -> dict:
        return {
            "name": self.name, "label": self.label, "kind": self.kind, "format": self.fmt,
            "dir": self.dir, "pattern": self.pattern, "task_key": self.task_key,
            "solver_field": self.solver_field, "status_rule": self.status_rule,
            "provenance": self.provenance, "seeds": f"{min(self.seeds)}-{max(self.seeds)}" if self.seeds else "",
            "n_seeds": len(self.seeds), "tasks": self.tasks,
            "known_missing": sorted(f"{t}:{s}" for t, s in self.known_missing),
            "required": self.required, "trace_len": self.trace_len, "budget": self.budget,
            "coldstart_from": self.coldstart_from, "note": self.note, "namespace": self.namespace,
            "cr_arm": self.cr_arm, "expect": self.expect, "display": self.display,
            "commit_root": self.commit_root, "fallback_rule": self.fallback_rule,
        }


@dataclass
class SourcesConfig:
    path: str
    sha256: str
    name: str
    description: str
    roots: dict
    tasks: list[str]
    seeds: list[int]
    cell_filter: str
    arms: dict[str, ArmSpec]
    includes: list[dict]
    seeds_subset: list[int] | None = None     # the command-line --seeds subset (None: every declared seed)

    def arm(self, name: str) -> ArmSpec:
        if name not in self.arms:
            raise ConfigError(f"arm {name!r} is not in the sources config {self.path}")
        return self.arms[name]


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def _arm_from_dict(name: str, a: dict, roots: dict, tasks: list[str], seeds: list[int],
                   namespace: str = "", seeds_subset: list[int] | None = None) -> ArmSpec:
    """``seeds`` = the config-level default seed list, already narrowed by the command-line subset;
    ``seeds_subset`` = that command-line subset (None when not given). An arm's own "seeds" entry is
    narrowed by the command-line subset only, never by the config default: the replication arms
    ('<arm>__s82', seeds 82-121) lie outside the default 42-81."""
    allowed = {"label", "kind", "format", "dir", "pattern", "task_key", "solver_field",
               "status_rule", "provenance", "seeds", "tasks", "known_missing", "required",
               "trace_len", "budget", "coldstart_from", "note", "cr_arm", "expect", "display",
               "wave", "commit_root", "fallback_rule"}
    bad = set(a) - allowed
    if bad:
        raise ConfigError(f"arm {name}: unknown keys {sorted(bad)}")
    for req in ("dir", "pattern", "format", "kind"):
        if req not in a:
            raise ConfigError(f"arm {name}: missing key {req!r}")
    if a["format"] not in ("ours", "external"):
        raise ConfigError(f"arm {name}: format must be ours|external")
    if a["kind"] not in ("cr", "stored"):
        raise ConfigError(f"arm {name}: kind must be cr|stored")
    arm_tasks = list(a.get("tasks") or tasks)
    arm_tasks = [t for t in arm_tasks if t in tasks]      # run subset (smoke) wins
    own_seeds = parse_seeds(a.get("seeds"))
    if own_seeds:
        arm_seeds = [s for s in own_seeds if seeds_subset is None or s in seeds_subset]
    else:
        arm_seeds = list(seeds)
    if len(set(arm_seeds)) != len(arm_seeds):
        raise ConfigError(f"arm {name}: duplicate seeds")
    m_root = re.match(r"^\$\{([A-Z0-9_]+)\}", a["dir"])
    commit_root = a.get("commit_root") or (m_root.group(1) if m_root else None)
    km = set()
    for c in a.get("known_missing", []):
        t, s = c.split(":")
        if t not in TASKS_T11:
            raise ConfigError(f"arm {name}: known_missing task {t!r} unknown")
        km.add((t, int(s)))
    km = {c for c in km if c[0] in arm_tasks and c[1] in arm_seeds}
    prov = a.get("provenance", "require" if a["kind"] == "cr" else "record")
    if a["kind"] == "cr" and prov != "require" and not namespace:
        raise ConfigError(f"arm {name}: a 'cr' arm must use provenance 'require'")
    spec = ArmSpec(
        name=(f"{namespace}:{name}" if namespace else name),
        label=a.get("label", name), kind=("posted" if namespace else a["kind"]), fmt=a["format"],
        dir=_subst(a["dir"], roots), pattern=a["pattern"], task_key=a.get("task_key", "poli"),
        solver_field=a.get("solver_field"),
        status_rule=a.get("status_rule", "require_field"),
        provenance=("record" if namespace else prov), seeds=arm_seeds, tasks=arm_tasks,
        known_missing=km, required=bool(a.get("required", True)),
        trace_len=int(a.get("trace_len", 30)), budget=a.get("budget", 30),
        coldstart_from=(f"{namespace}:{a['coldstart_from']}" if namespace and a.get("coldstart_from")
                        else a.get("coldstart_from")),
        note=a.get("note", ""), namespace=namespace,
        cr_arm=(None if namespace else a.get("cr_arm")), expect=dict(a.get("expect") or {}),
        display=dict(a.get("display") or {}),
        commit_root=(None if namespace else commit_root), declared_seeds=bool(own_seeds),
        fallback_rule=a.get("fallback_rule", "surrogate_fallback_count"),
    )
    if spec.fallback_rule not in FALLBACK_RULES:
        raise ConfigError(f"arm {name}: fallback_rule must be one of {FALLBACK_RULES}")
    if set(spec.display) - {"label", "support"}:
        raise ConfigError(f"arm {name}: display keys must be label/support")
    if spec.status_rule not in ("require_field", "absent_is_completed"):
        raise ConfigError(f"arm {name}: bad status_rule {spec.status_rule}")
    if spec.fmt == "external" and not spec.coldstart_from:
        raise ConfigError(f"arm {name}: external format needs coldstart_from")
    spec.build_regex()
    return spec


def load_config(path: str, root_overrides: dict | None = None, tasks_subset: list[str] | None = None,
                seeds_subset: list[int] | None = None, load_includes: bool = True) -> SourcesConfig:
    path = os.path.abspath(path)
    with open(path) as f:
        raw = json.load(f)
    if raw.get("schema") != SCHEMA:
        raise ConfigError(f"{path}: schema must be {SCHEMA!r}")
    roots = dict(raw.get("roots", {}))
    roots.update(root_overrides or {})
    tasks = list(raw.get("tasks") or TASKS_T11.keys())
    for t in tasks:
        if t not in TASKS_T11:
            raise ConfigError(f"unknown task {t}")
    seeds = parse_seeds(raw.get("seeds", "42-81"))
    if tasks_subset:
        tasks = [t for t in tasks if t in tasks_subset]
    seeds_subset = list(seeds_subset) if seeds_subset else None
    if seeds_subset:
        seeds = [s for s in seeds if s in seeds_subset]
    cell_filter = raw.get("cell_filter", "completed")
    if cell_filter not in ("completed", "clean"):
        raise ConfigError("cell_filter must be completed|clean")
    arms: dict[str, ArmSpec] = {}
    for name, a in raw["arms"].items():
        if name.startswith("_"):
            continue
        arms[name] = _arm_from_dict(name, a, roots, tasks, seeds, seeds_subset=seeds_subset)
    includes = []
    if load_includes:
        for inc in raw.get("include", []):
            ns = inc["namespace"]
            ipath = os.path.join(os.path.dirname(path), inc["file"])
            with open(ipath) as f:
                iraw = json.load(f)
            iroots = dict(iraw.get("roots", {}))
            iroots.update({k: v for k, v in (root_overrides or {}).items() if k in iroots and k != "CR"})
            want = inc.get("arms", "all")
            for name, a in iraw["arms"].items():
                if name.startswith("_") or (want != "all" and name not in want):
                    continue
                a2 = copy.deepcopy(a)
                a2["required"] = bool(a.get("required", True)) and not inc.get("optional", False)
                arms[f"{ns}:{name}"] = _arm_from_dict(name, a2, iroots, tasks, seeds, namespace=ns,
                                                      seeds_subset=seeds_subset)
            includes.append({"namespace": ns, "file": ipath, "sha256": _file_sha256(ipath)})
    for spec in arms.values():
        if spec.coldstart_from and spec.coldstart_from not in arms:
            raise ConfigError(f"{spec.name}: coldstart_from {spec.coldstart_from} is not an arm")
    return SourcesConfig(path=path, sha256=_file_sha256(path), name=raw.get("name", ""),
                         description=raw.get("description", ""), roots=roots, tasks=tasks,
                         seeds=seeds, cell_filter=cell_filter, arms=arms, includes=includes,
                         seeds_subset=seeds_subset)
