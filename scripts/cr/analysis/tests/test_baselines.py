"""Appendix-H facts table (appendixH_facts.csv) and tab_baselines: the audited block is untouched, every
row added for GEN v3 carries file:line evidence at the control tag, LaTeX escaping, the proofreading
markers, the cells computed from the loaded files, and a compile at the NeurIPS text width.

The proofread text (appendixH_printed.csv, 2026-10-22) replaces the facts-table text field by field. The tests of
the facts-table rendering run without it (autouse fixture ``_facts_only``); the tests at the end use it."""
import csv
import json
import os
import re
import shutil
import subprocess

import pytest

from conftest import arm_cfg, ours_record
from crlib import TASKS_T11, baselines
from crlib.config import load_config
from crlib.context import VARIANTS, Ctx
from crlib.engine import Engine
from crlib.latex import PREAMBLE_CHECK
from crlib.loader import load_all
from crlib.numbers import Registry

GRID = [(t, s) for t in ("adip", "rano") for s in (42, 43, 44)]
VERSIONS = {"numpy": "2.4.4", "torch": "2.11.0", "botorch": "0.17.2", "gpytorch": "1.15.2", "scipy": "1.17.1",
            "cma": "4.4.4", "selfies": "2.2.0", "linear_operator": "0.6.1"}
NEW_ARMS = ["papenmeier_msr_published", "turbo_box_faithful", "hvarfner_scaled_gp_box_argmax_raw",
            "hvarfner_scaled_gp_boxdir_argmax", "random_boxdir", "random_box_faithful", "geowalk_one_fifth",
            "hvarfner_scaled_gp_sphere05", "hvarfner_scaled_gp_sphere15", "hvarfner_scaled_gp_sphere20",
            # not requested, added because the camera-ready manifest runs them and the audit had no row
            "geowalk_const060", "geowalk_const035", "geowalk_const015", "geowalk_explore8", "geowalk_explore24",
            "cma_es_subspace_sd8", "cma_es_subspace_sd64", "random_search_prior",
            "hvarfner_scaled_gp_norm_box_argmax", "papenmeier_msr_published_norm_box"]


def _write_arm(tmp_path, dirname, solver, cells=GRID, versions=VERSIONS, no_versions=(), fallback=None):
    d = tmp_path / dirname
    d.mkdir(parents=True, exist_ok=True)
    for t, s in cells:
        rec = ours_record(solver, t, s, fallback=(fallback or {}).get((t, s), 0))
        if versions is not None and (t, s) not in no_versions:
            rec["provenance"].update(package_versions=dict(versions), python_version="3.12.14", cuda_version="13.0")
        (d / f"ours_{solver}_{TASKS_T11[t]}_s{s}_128d.json").write_text(json.dumps(rec))


REAL_PRINTED_CSV = baselines.PRINTED_CSV


@pytest.fixture(autouse=True)
def _facts_only(tmp_path, monkeypatch):
    """The facts-table rendering: no proofread text (the tests of the proofread text restore it)."""
    monkeypatch.setattr(baselines, "PRINTED_CSV", str(tmp_path / "no_printed_text.csv"))


@pytest.fixture
def printed(monkeypatch):
    monkeypatch.setattr(baselines, "PRINTED_CSV", REAL_PRINTED_CSV)
    return REAL_PRINTED_CSV


@pytest.fixture(autouse=True)
def _verdicts(tmp_path, monkeypatch):
    """A pinned synthetic verdict file: camera-ready card titles carry the printed names (second set (c))."""
    from v5_fixtures import write_verdicts
    p = tmp_path / "verification" / "verdicts.json"
    write_verdicts(p, monkeypatch=monkeypatch)
    return p


def _ctx(tmp_path, write_config, arms: dict, mode="cr"):
    ds = load_all(load_config(write_config(arms)), "abc1234", log=lambda m: None)
    return Ctx(ds=ds, eng=Engine(ds), reg=Registry("x", "0" * 64, "default"), var=VARIANTS["default"],
               outdir=str(tmp_path), script_commit="x", mode=mode,
               extra={"_shared_cache": {}, "verdicts_path": str(tmp_path / "verification" / "verdicts.json")})


# ---- the facts table --------------------------------------------------------------------------------
def test_audited_block_is_untouched_and_guarded(tmp_path):
    rows = baselines.load_facts()
    assert len(rows) == baselines.N_AUDITED_ROWS + 13 + 14 + 2 + 1  # + 14 published-recipe rows of GEN v5, + the C4
    #                                                                  row of 2026-11-06 (GEN v7)
    # + 2 rows of the GuacaMol runs of 2026-11-02 (root CRRD): full-dimensional CMA-ES (two populations), BAxUS
    assert sum(1 for r in rows if not r["_new"]) == 54 and [r["arm"] for r in rows[:3]] == \
        ["geowalk", "cma_es_subspace", "random_search"]
    raw = list(csv.reader(open(baselines.FACTS_CSV, newline="", encoding="utf-8")))
    assert raw[0] == baselines.AUDITED_COLUMNS + baselines.APPENDED_COLUMNS
    k = len(baselines.AUDITED_COLUMNS)
    assert baselines.audited_block_sha256(raw[0][:k], [r[:k] for r in raw[1:55]]) == baselines.AUDITED_BLOCK_SHA256
    # one changed audited cell is refused
    raw[7][3] = raw[7][3] + " "
    bad = tmp_path / "facts.csv"
    with open(bad, "w", newline="", encoding="utf-8") as f:
        csv.writer(f, lineterminator="\n").writerows(raw)
    with pytest.raises(RuntimeError, match="audited cell was changed"):
        baselines.load_facts(str(bad))
    # an appended cell of an audited row may be edited
    raw[7][3] = raw[7][3][:-1]
    raw[7][k + 1] = "edited by an author"
    with open(bad, "w", newline="", encoding="utf-8") as f:
        csv.writer(f, lineterminator="\n").writerows(raw)
    assert baselines.load_facts(str(bad))[6]["lengthscale initialisation"] == "edited by an author"


def test_new_rows_cover_the_new_arms_with_code_evidence():
    rows = [r for r in baselines.load_facts() if r["_new"] and r["row origin"].startswith(baselines.ORIGIN_NEW)]
    assert len(rows) == 13          # the GEN v3 rows (the GEN v5 rows are tested in test_v5.py)
    covered = [a for r in rows for a in r["_run"]]
    assert set(NEW_ARMS) <= set(covered)
    line_ref = re.compile(r"(src/spherevo/(gp_competitors|solver|turbo_box_faithful)\.py|scripts/run_benchmark\.py):\d+")
    for r in rows:
        code = r["implementing code (abbd61e unless stated)"]
        assert line_ref.search(code), r["arm"]
        assert "cr-controls-2026-10-12" in code and "scripts/cr/manifest.json:" in code, r["arm"]
        assert "scripts/run_benchmark.py:" in code, r["arm"]             # the registration is cited too
        assert all(r[c].strip() for c in baselines.AUDITED_COLUMNS), r["arm"]
        assert r["needs proofreading"] == "all" and r["row origin"].startswith(baselines.ORIGIN_NEW)
        assert "cr-controls-2026-10-12" in r["evidence"]
        assert r["lengthscale initialisation"] and r["lengthscale prior"]


def test_replication_arms_are_listed_with_their_base_row_and_nowhere_else():
    rows = baselines.load_facts()
    s82 = {a: r["arm"] for r in rows for a in r["_run"] if a.endswith("__s82")}
    assert s82 == {b + "__s82": b for b in ("random_search", "random_box_faithful", "hvarfner_scaled_gp",
                                             "hvarfner_scaled_gp_box_argmax", "hvarfner_scaled_gp_sphere_continuous",
                                             "hvarfner_scaled_gp_box_faithful")}
    assert all(not r["_run"] for r in rows if r["arm"].startswith(("lassobench:", "nasbench:")))
    ov = {r["arm"]: sorted(r["_overrides"]) for r in rows if r["_overrides"]}
    assert ov == {a: ["deviations", "kernel", "lengthscale initialisation"]
                  for a in ("papenmeier_msr_norm_box", "papenmeier_msr_box")}


# ---- LaTeX escaping ---------------------------------------------------------------------------------
def test_tex_escape_specials_and_break_points():
    e = baselines.tex_escape
    assert e("a_b 50% x^2 {p01} <1e-6 |x| A&B #1 ~0.47") == \
        "a\\_b 50\\% x\\^{}2 \\{p01\\} \\textless{}1e-6 \\textbar{}x\\textbar{} A\\&B \\#1 \\textasciitilde{}0.47"
    assert e("--no-noiseless") == "-{}-no-noiseless"
    long = e("results/guacamol/spherevo/ours_random_search_*_128d.json:")
    assert long.count("\\allowbreak{}") >= 6 and "\\_\\allowbreak{}" in long
    assert "\\allowbreak" not in e("short_token x")                    # short tokens are left alone
    # no cell of the facts table has a stretch TeX could not break inside the text column
    worst = max(baselines.longest_unbreakable(v) for r in baselines.load_facts()
                for k, v in r.items() if isinstance(v, str))
    assert worst <= 60, worst
    for r in baselines.load_facts():
        for k, v in r.items():
            if isinstance(v, str):
                v.encode("ascii")                                       # escaping covers ASCII only
                assert "\n" not in v


# ---- the table on a synthetic run --------------------------------------------------------------------
def _synthetic(tmp_path, write_config, kind="cr"):
    arms = {}
    for name in ("geowalk", "hvarfner_scaled_gp", "papenmeier_msr_norm_box", "papenmeier_msr_published",
                 "turbo_box_faithful", "hvarfner_scaled_gp_sphere05", "hvarfner_scaled_gp_sphere15",
                 "geowalk_flat"):
        _write_arm(tmp_path, name, name, no_versions={("rano", 44)} if name == "geowalk" else ())
        arms[name] = arm_cfg(name, kind=kind)
    _write_arm(tmp_path, "random_search", "random_search")
    _write_arm(tmp_path, "random_search__s82", "random_search", cells=[("adip", 42), ("rano", 42)])
    arms["random_search"] = arm_cfg("random_search", kind=kind)
    arms["random_search__s82"] = arm_cfg("random_search", kind=kind, dir="${T}/random_search__s82",
                                         known_missing=["adip:43", "adip:44", "rano:43", "rano:44"])
    return _ctx(tmp_path, write_config, arms, mode="cr" if kind == "cr" else "legacy")


def _code(tex: str) -> str:
    return tex.replace("\\allowbreak{}", "").replace("\\_", "_").replace("\\{", "{").replace("\\}", "}")


def _fields(t, titles=None):
    """{facts-row arm: {field: text}} keyed by the card's code-name field (the title carries the printed label)."""
    cards, cur, title = {}, None, None
    for r in t.rows:
        if r[0] == "card":
            title, cur = r[1], None
        elif r[0] == "field" and r[1] == "Code name":
            tt = r"\\texttt\{((?:\\[{}]|\{\}|[^{}])*)\}"
            mo = re.search("facts row " + tt, r[2]) or re.search(tt, r[2])
            cur = _code(mo.group(1))
            cards[cur] = {}
            if titles is not None:
                titles[cur] = (title, r[2])
        elif r[0] == "field":
            cards[cur][r[1]] = r[2]
    return cards


def test_table_on_camera_ready_arms(tmp_path, write_config):
    ctx = _synthetic(tmp_path, write_config)
    t = baselines.tab_baselines(ctx)
    cards = _fields(t)
    rec = ctx.reg.records
    names = [n for _, n, _ in baselines.FIELDS]
    assert len(names) == 10 and all(list(c) == names for c in cards.values())      # the ten promised fields
    # GuacaMol cards of the present arms + the 7 LassoBench, 8 LassoBench-published and 6 NAS rows of the facts table
    assert {"geowalk", "random_search", "hvarfner_scaled_gp", "papenmeier_msr_norm_box", "papenmeier_msr_published",
            "turbo_box_faithful", "hvarfner_scaled_gp_sphere{05,15,20}"} <= set(cards)
    assert "papenmeier_msr" not in cards and rec["crBaselinesCards"]["value"] == 7 + 13 + 8
    # software version and seed count of a camera-ready arm come from the files (double dagger)
    sw = cards["turbo_box_faithful"]["Software version"]
    assert "torch 2.11.0 (CUDA 13.0), botorch 0.17.2" in sw and "(6 of 6 result files)" in sw
    assert sw.endswith(baselines.DDAGGER) and "py3.9" not in sw
    assert "2 environments in 6 result files" in cards["geowalk"]["Software version"]
    assert "no package versions recorded" in cards["geowalk"]["Software version"]
    assert rec["crBaselinesGeowalkEnvSets"]["value"] == 2 and rec["crBaselinesTurboBoxFaithfulEnvSets"]["value"] == 1
    sd = cards["random_search"]["Seed count"]
    assert "3 seeds (42--44) $\\times$ 2 tasks: 6 result files, 6 completed" in sd
    assert "(b) 1 seed (42) $\\times$ 2 tasks" in sd and "\\texttt" not in sd
    assert "Audit of the stored files" not in sd                      # stored-file counts are not mixed in
    assert rec["crBaselinesRandomSearchSEightTwoSeeds"]["value"] == 1
    assert rec["crBaselinesRandomSearchCells"]["files"]["count"] == 6
    # grouped row: both loaded radius arms, the absent one is recorded
    titles = {}
    _fields(t, titles)
    title, code = titles["hvarfner_scaled_gp_sphere{05,15,20}"]
    assert "(a) \\texttt{hvarfner\\_scaled\\_gp\\_sphere05}" in _code(code).replace("_", "\\_")
    assert cards["hvarfner_scaled_gp_sphere{05,15,20}"]["Seed count"].startswith("(a) 3 seeds")
    # second set (c): titles carry printed labels, the code names only the code-name field
    assert all("\\texttt" not in ti for ti, _ in titles.values())
    # the cards of the other benchmarks are titled by their printed labels, not by the facts-row code name
    assert any(r.startswith(("lassobench:", "lassobench_pub:", "nasbench:")) for r in titles)
    assert not any(("lassobench" in ti or "nasbench" in ti) for ti, _ in titles.values())
    assert titles["papenmeier_msr_published"][0].startswith("\\textbf{Mat\\'ern-5/2 GP ($\\ell_0=0.1\\sqrt{D}$), LogEI")
    assert not any("\\texttt" in v for c in cards.values() for k, v in c.items() if k in ("Seed count", "Software version"))
    assert any(s["what"] == "hvarfner_scaled_gp_sphere20" for s in ctx.skipped)
    # camera-ready override replaces the stored-file description and is marked
    ini = cards["papenmeier_msr_norm_box"]["Lengthscale initialization"]
    assert ini.startswith("0.1 sqrt(D) = 1.6") and "Audited description of the stored files" in ini
    assert ini.endswith(baselines.DAGGER)
    # verbatim audited cell, no marker; uncertain audited text is marked; new rows are marked everywhere
    acq = cards["hvarfner_scaled_gp"]["Acquisition"]
    assert acq.startswith("qLogExpectedImprovement (MC; BoTorch default sampler)") and "; maximiser: argmax over 2048" in acq
    assert not acq.endswith(baselines.DAGGER)
    assert cards["hvarfner_scaled_gp"]["Lengthscale prior"].endswith(baselines.DAGGER)          # derived cell
    pub = cards["papenmeier_msr_published"]
    assert all(v.endswith((baselines.DAGGER, baselines.DDAGGER)) for v in pub.values())
    assert sum(v.endswith(baselines.DDAGGER) for v in pub.values()) == 2
    assert "LogExpectedImprovement" in pub["Acquisition"] and "num\\_restarts=5" in pub["Acquisition"].replace("\\allowbreak{}", "")
    assert cards["lassobench:hvarfner_gp"]["Software version"].endswith(baselines.DAGGER)
    # marker counts are the registered numbers
    cells = [v for c in cards.values() for v in c.values()]
    assert rec["crBaselinesCells"]["value"] == len(cells) == 10 * 28
    assert rec["crBaselinesProofCells"]["value"] == sum(v.endswith(baselines.DAGGER) for v in cells)
    assert rec["crBaselinesComputedCells"]["value"] == sum(v.endswith(baselines.DDAGGER) for v in cells) == 2 * 7
    assert rec["crBaselinesMarkedCells"]["value"] == rec["crBaselinesProofCells"]["value"] + 14
    # an arm of the run without a facts row is reported, never dropped silently
    assert rec["crBaselinesArmsWithoutRow"]["value"] == 1
    assert any(s["what"] == "geowalk_flat" and "no Appendix-H facts row" in s["why"] for s in ctx.skipped)
    assert all(r["files"]["count"] > 0 for r in rec.values())


def test_table_on_stored_arms_keeps_the_audited_text(tmp_path, write_config):
    ctx = _synthetic(tmp_path, write_config, kind="stored")
    t = baselines.tab_baselines(ctx)
    cards = _fields(t)
    sw = cards["hvarfner_scaled_gp"]["Software version"]
    assert sw.startswith("CUDA; py3.12.x")                              # the audited text comes first, verbatim
    assert "Recorded in the loaded files:" in sw and "torch 2.11.0 (CUDA 13.0)" in sw and "(6 of 6 result files)" in sw
    assert sw.endswith(baselines.DAGGER)                              # the audited text says 'NOT recorded'
    sd = cards["hvarfner_scaled_gp"]["Seed count"]
    assert "3 seeds (42--44)" in sd and "Audit of the stored files" in sd and "spherevo: 440 files" in sd
    assert sd.endswith(baselines.DDAGGER)
    # a stored arm under a row written for this version: the row's text is marked, the files are still read
    sp = cards["hvarfner_scaled_gp_sphere{05,15,20}"]
    # machines are named by their hardware (GEN v5 backlog item 1), never by host name
    assert "stored generation: server C" in sp["Software version"] and sp["Software version"].endswith(baselines.DAGGER)
    assert not any(h in v for c in cards.values() for v in c.values() for h in ("tulen", "kajman", "utia.cas.cz"))
    assert sp["Seed count"].endswith(baselines.DAGGER)
    ini = cards["papenmeier_msr_norm_box"]["Lengthscale initialization"]
    assert ini.startswith("sqrt(D) = 16") and "Audited description" not in ini      # no camera-ready override
    assert "No camera-ready run is loaded here" in t.caption
    assert ctx.reg.records["crBaselinesHvarfnerScaledGpEnvSets"]["value"] == 1


def test_no_guacamol_card_gives_no_table(tmp_path, write_config):
    _write_arm(tmp_path, "geowalk_flat", "geowalk_flat")
    ctx = _ctx(tmp_path, write_config, {"geowalk_flat": arm_cfg("geowalk_flat")})
    assert baselines.tab_baselines(ctx) is None and not ctx.reg.records
    assert any(s["table"] == "tab_baselines" for s in ctx.skipped)


@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="pdflatex not available")
def test_table_compiles_at_text_width_without_overfull_boxes(tmp_path, write_config):
    ctx = _synthetic(tmp_path, write_config)
    t = baselines.tab_baselines(ctx)
    out = tmp_path / "out"
    out.mkdir()
    t.write(str(out), "test")
    ctx.reg.write_tex(str(out / "numbers.tex"), "test")
    (out / "check.tex").write_text(PREAMBLE_CHECK + "\\input{numbers.tex}\n\\input{tab_baselines.tex}\n\\end{document}\n")
    for _ in range(2):          # longtable needs two passes
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "check.tex"], cwd=out,
                           capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout[-2000:]
    log = (out / "check.log").read_text(errors="replace")
    assert "Overfull" not in log, [ln for ln in log.splitlines() if "Overfull" in ln][:5]
    assert not [ln for ln in log.splitlines() if ln.startswith("!")]
    assert os.path.getsize(out / "check.pdf") > 10_000


def test_effective_support_json_was_built_from_this_facts_table():
    """effective_support.json records the sha256 of the CSV it was built from; after an edit of the CSV rerun
    ``python tools/build_effective_support.py appendixH_facts.csv effective_support.json``."""
    import hashlib
    here = os.path.dirname(baselines.FACTS_CSV)
    eff = json.load(open(os.path.join(here, "effective_support.json")))
    assert eff["source_csv"] == "appendixH_facts.csv"
    assert eff["source_csv_sha256"] == hashlib.sha256(open(baselines.FACTS_CSV, "rb").read()).hexdigest()
    arms = [e["arm"] for e in eff["arms"]]
    assert len(arms) == len(set(arms)) == 74        # 2026-11-06: + hvarfner_scaled_gp_sphere_lbfgs (control C4)
    assert {"hvarfner_scaled_gp_box_argmax_raw", "hvarfner_scaled_gp_boxdir_argmax", "random_boxdir"} <= set(arms)
    assert not [a for a in arms if a.endswith("__s82")]            # replication arms have no row of their own
    facts_rows = {r["arm"] for r in baselines.load_facts()}
    for e in eff["arms"]:
        if e.get("facts"):
            assert e["source"].split("'")[1] in facts_rows, e["arm"]


def test_arm_reading_substitute_files_is_left_out_of_its_card(tmp_path, write_config):
    """Legacy mode points papenmeier_msr_published at the July RAASP files (display override): the published-MSR
    facts row must not describe them."""
    _write_arm(tmp_path, "spherevo", "papenmeier_msr_raasp")
    _write_arm(tmp_path, "norm_archive", "papenmeier_msr_norm_box")
    arms = {
        "papenmeier_msr_raasp": arm_cfg("papenmeier_msr_raasp", kind="stored", dir="${T}/spherevo"),
        "papenmeier_msr_published": arm_cfg("papenmeier_msr_raasp", kind="stored", dir="${T}/spherevo",
                                            display={"label": "Papenmeier RAASP trust-region variant"}),
        # a relabelled arm that reads its OWN stored files keeps its card
        "papenmeier_msr_norm_box": arm_cfg("papenmeier_msr_norm_box", kind="stored", dir="${T}/norm_archive",
                                           display={"label": "Papenmeier, sqrt(D) archive"}),
    }
    ctx = _ctx(tmp_path, write_config, arms, mode="legacy")
    cards = _fields(baselines.tab_baselines(ctx))
    assert "papenmeier_msr_published" not in cards
    assert {"papenmeier_msr_raasp", "papenmeier_msr_norm_box"} <= set(cards)
    assert any(s["what"] == "papenmeier_msr_published" and "substitute files" in s["why"] for s in ctx.skipped)
    assert "crBaselinesPapenmeierMsrPublishedSeeds" not in ctx.reg.records
    assert ctx.reg.records["crBaselinesArmsWithoutRow"]["value"] == 0          # it has a row, only no card
    assert not [s for s in ctx.skipped if "no Appendix-H facts row" in s["why"]]


def test_card_mixing_a_camera_ready_arm_and_a_stored_arm(tmp_path, write_config):
    _write_arm(tmp_path, "random_box_faithful", "random_box_faithful")
    _write_arm(tmp_path, "p1", "random_box_faithful", versions={"numpy": "2.4.4", "torch": "2.11.0"})
    arms = {"random_box_faithful": arm_cfg("random_box_faithful"),
            "p1_random_box_faithful": arm_cfg("random_box_faithful", kind="stored", dir="${T}/p1")}
    ctx = _ctx(tmp_path, write_config, arms)
    card = _fields(baselines.tab_baselines(ctx))["random_box_faithful"]
    sw = card["Software version"].replace("\\allowbreak{}", "")
    assert sw.startswith("python 3.12.14, torch 2.11.0 (CUDA 13.0), botorch 0.17.2")          # the camera-ready files
    assert "\\emph{Stored arms (b), facts table:}" in sw and "\\texttt" not in sw
    assert "\\emph{Recorded in the loaded files:} python 3.12.14, torch 2.11.0 (CUDA 13.0), numpy 2.4.4; host" in sw
    assert sw.endswith(baselines.DAGGER)                    # the row was written for this version
    sd = card["Seed count"].replace("\\allowbreak{}", "")
    assert "(a) 3 seeds" in sd and "(b) 3 seeds" in sd and "\\texttt" not in sd
    assert "Audit of the stored files" in sd
    assert ctx.reg.records["crBaselinesPOneRandomBoxFaithfulCells"]["value"] == 6


# ---- the proofread text -----------------------------------------------------------------------------
PRINTED_ROWS_ARMS = ("geowalk", "hvarfner_scaled_gp", "papenmeier_msr_published", "turbo_box_faithful", "gabo")


def _by_row(t):
    """{facts-row arm: (title, {field: text})} from the card rows (the row arm rides along in the card tuple)."""
    out, cur = {}, None
    for r in t.rows:
        if r[0] == "card":
            cur = r[2]
            out[cur] = (r[1], {})
        elif r[0] == "field":
            out[cur][1][r[1]] = r[2]
    return out


def test_printed_text_is_complete_clean_and_sourced(printed):
    rows = baselines.load_facts()
    P = baselines.load_printed(printed, rows)
    by_row = {}
    for (row, field), e in P.items():
        by_row.setdefault(row, set()).add(field)
        assert e["source"].strip() and (e["text"] or e["verdict"] == "omitted")
    facts = {r["arm"]: r for r in rows}
    eight = {k for k, _, _ in baselines.FIELDS} - {"software", "seeds"}
    for row, fields in by_row.items():
        assert eight <= fields, (row, eight - fields)
        if not facts[row]["_run"]:                   # other benchmark: every field is text (and a title, except for
            want = {k for k, _, _ in baselines.FIELDS}  # published methods, whose title is the verified name)
            assert want | ({"title"} if not row.startswith("lassobench_pub:") else set()) == fields, row
    # every card of the final camera-ready run is covered (64 cards: the GuacaMol rows of the printed arms,
    # 8 LassoBench published, 7 LassoBench rerun, 6 NAS-Bench-101)
    assert len(by_row) == 67            # + the two cards of the GuacaMol runs of 2026-11-02 (root CRRD), + the C4 card
    #                                     of 2026-11-06 (RUN_RULES 2026-11-06 16:30)
    assert sum(1 for r in by_row if r.startswith("lassobench:")) == 7
    assert sum(1 for r in by_row if r.startswith("lassobench_pub:")) == 8
    assert sum(1 for r in by_row if r.startswith("nasbench:")) == 6
    # facts that were wrong in the facts table and are corrected (audit 2026-10-20 and proofreading 2026-10-22)
    assert "0.5 in every dimension" in P[("turbo_published", "lengthscale initialisation")]["text"]
    assert "[-8, 8]^256" in P[("linear_bo_box8", "support")]["text"]
    assert "rescaled latent box" not in P[("linear_bo_box8", "deviations")]["text"]
    assert P[("xu_published", "lengthscale prior")]["text"].startswith("Uniform(1e-10, 30)")
    assert "0.001" not in P[("xu_published", "lengthscale prior")]["text"]
    cow = P[("cowboys", "deviations")]["text"]
    assert "Crank-Nicolson" in cow and "not printed" in cow
    assert "x0 = 0" in P[("lassobench_pub:cmaes_full_published", "deviations")]["text"]
    assert "own 5-point" in P[("lassobench_pub:baxus_official", "deviations")]["text"]
    assert "{fallback_steps}" in P[("gabo", "acquisition")]["text"]
    assert "not spherical" in P[("lassobench:cmaes_sub", "deviations")]["text"]


def test_printed_text_refuses_internal_text(tmp_path, printed):
    raw = list(csv.reader(open(printed, newline="", encoding="utf-8")))
    rows = baselines.load_facts()
    for bad in ("see /data2/prusek/x", "UNVERIFIED here", "per D-X2", "solver.py:78-80", "maximiser", "learning_rte",
                "wrapper cites Stanton et al.", "decision D2 of H2_successor"):
        r2 = [list(x) for x in raw]
        r2[1][3] = bad
        p = tmp_path / "bad.csv"
        with open(p, "w", newline="", encoding="utf-8") as f:
            csv.writer(f, lineterminator="\n").writerows(r2)
        with pytest.raises(RuntimeError, match="forbidden text"):
            baselines.load_printed(str(p), rows)
    r2 = [list(x) for x in raw] + [list(raw[1])]
    with open(tmp_path / "dup.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f, lineterminator="\n").writerows(r2)
    with pytest.raises(RuntimeError, match="duplicate"):
        baselines.load_printed(str(tmp_path / "dup.csv"), rows)


def test_table_with_printed_text(tmp_path, write_config, printed):
    arms = {}
    for name in PRINTED_ROWS_ARMS:
        fb = {("adip", 42): 3, ("rano", 43): 1} if name == "gabo" else None
        _write_arm(tmp_path, name, name, fallback=fb)
        arms[name] = arm_cfg(name)
    ctx = _ctx(tmp_path, write_config, arms)
    t = baselines.tab_baselines(ctx)
    cards = _by_row(t)
    cells = [v for _, c in cards.values() for v in c.values()]
    # no dagger anywhere: every printed field of these cards is proofread
    assert not any(v.endswith(baselines.DAGGER) for v in cells)
    assert ctx.reg.records["crBaselinesProofCells"]["value"] == 0
    assert baselines.DAGGER not in t.caption and "facts table" not in t.caption and "abbd61e" not in t.caption
    assert "Table~\\ref{tab:verification}" in t.caption
    # the computed fields stay computed
    assert cards["geowalk"][1]["Software version"].endswith(baselines.DDAGGER)
    assert cards["geowalk"][1]["Seed count"].endswith(baselines.DDAGGER)
    # proofread text, American spelling, no internal paths or codes
    gw = cards["geowalk"][1]
    assert gw["Repository"] == "this repository (method of this paper)"
    assert "maximizer" in gw["Acquisition"] and "maximiser" not in " ".join(cells)
    assert list(gw) == ["Code name"] + [n for _, n, _ in baselines.FIELDS]
    assert not any(x in " ".join(cells) for x in ("/data2", "/home/prusek", "UNVERIFIED", "Stanton", "learning_rte",
                                                    "H2_successor", "facts row"))
    # count tokens are filled from the loaded files
    acq = cards["gabo"][1]["Acquisition"]
    assert "4 such steps in 2 of the 6 runs" in acq and "{" not in acq.replace("\\{", "")
    assert ctx.reg.records["crBaselinesGaboFallbackSteps"]["value"] == 4
    # other benchmarks: titles from the proofread text, cross-references survive the escaping
    titles = {r: ti for r, (ti, _) in cards.items()}
    assert titles["lassobench:hvarfner_gp"] == "\\textbf{RBF GP (DSP prior), qLogNEI, L-BFGS-B (rerun)}"
    assert "(stored runs)" not in titles["lassobench:geowalk"]
    assert "Table~\\ref{tab:lassobench}" in cards["lassobench:geowalk"][1]["Seed count"]
    assert cards["lassobench:geowalk"][1]["Code name"].replace("\\allowbreak{}", "") == \
        "\\texttt{geowalk} in \\texttt{scripts/run\\_lassobench.py}"
    assert not any(v.endswith(baselines.DAGGER) for r, (_, c) in cards.items() if ":" in r for v in c.values())
    # group headings carry no author or method names
    from crlib import labels
    heads = [r[1] for r in t.rows if r[0] == "group"]
    assert heads and all(tok not in h for h in heads for tok in labels.AUTHOR_TOKENS)
    assert "\\textbf{Field} & \\textbf{Description}" in t.render("x")


@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="pdflatex not available")
def test_printed_table_compiles_at_text_width(tmp_path, write_config, printed):
    arms = {}
    for name in PRINTED_ROWS_ARMS:
        _write_arm(tmp_path, name, name)
        arms[name] = arm_cfg(name)
    ctx = _ctx(tmp_path, write_config, arms)
    t = baselines.tab_baselines(ctx)
    out = tmp_path / "out"
    out.mkdir()
    t.write(str(out), "test")
    ctx.reg.write_tex(str(out / "numbers.tex"), "test")
    (out / "check.tex").write_text(PREAMBLE_CHECK + "\\providecommand{\\geowalk}{GeoWalk}\\providecommand{\\geocma}{GeoCMA}\n"
                                   "\\input{numbers.tex}\n\\input{tab_baselines.tex}\n\\end{document}\n")
    for _ in range(2):
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "check.tex"], cwd=out,
                           capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout[-2000:]
    log = (out / "check.log").read_text(errors="replace")
    assert "Overfull" not in log, [ln for ln in log.splitlines() if "Overfull" in ln][:5]
