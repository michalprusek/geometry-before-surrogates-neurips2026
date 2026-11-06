"""RUN_RULES 2026-10-24: the Appendix H overview, the repository records of the cards and the run status, the settings
table of Appendix A (tab_hyper) and the compute table of Appendix K (tab_compute)."""
import csv
import hashlib
import importlib.util
import inspect
import json
import os
import re
import shutil
import sys
import types

import pytest

from crlib import baselines, compute, hyper, labels, overview, records, runtime
from crlib.baselines import CardTable
from crlib.latex import Table

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYSIS = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(os.path.dirname(ANALYSIS)))


# =====================================================================================================
# overview: appendixH_overview.csv
# =====================================================================================================
def _rows():
    with open(overview.OVERVIEW_CSV, newline="", encoding="utf-8") as f:
        return list(csv.reader(f))


def _write(tmp_path, rows):
    p = tmp_path / "ov.csv"
    with open(p, "w", newline="", encoding="utf-8") as f:
        csv.writer(f, lineterminator="\n").writerows(rows)
    return str(p)


def test_overview_csv_passes_every_rule_and_covers_every_facts_row():
    rows = overview.load_overview()
    facts = baselines.load_facts()
    covered = {c for r in rows for c, _ in r["_cards"]}
    assert covered | set(overview.OVERVIEW_OMITTED) == {r["arm"] for r in facts}
    assert not covered & set(overview.OVERVIEW_OMITTED)
    assert all(reason.strip() for reason in overview.OVERVIEW_OMITTED.values())
    # every arm that a camera-ready run does not load is omitted, and only those or rows printed nowhere
    from crlib import scope
    for r in facts:
        if r["_run"] and all(a in scope.NOT_LOADED_ARMS for a in r["_run"]):
            assert r["arm"] in overview.OVERVIEW_OMITTED, r["arm"]
    assert 24 <= len(rows) <= 32          # 24 since RUN_RULES 2026-11-06 21:30 (no TuRBO-1 re-implementation row)
    for r in rows:
        for col in overview.WRITTEN:
            assert overview.visible_length(r[col]) <= overview.MAX_CELL
            assert not baselines.PRINTED_FORBIDDEN.search(r[col])


def test_overview_has_the_rows_the_paper_points_to():
    rows = {r["row"]: r for r in overview.load_overview()}
    cards = {r["row"]: {c if a is None else f"{c}@{a}" for c, a in r["_cards"]} for r in rows.values()}
    hp = "hvarfner_published / hvarfner_published_allobs"
    # the second run of the Hvarfner recipe (tab_published: "Appendix H describes it") has a row of its own
    assert any(f"{hp}@hvarfner_published_allobs" in c for c in cards.values())
    assert any(f"{hp}@hvarfner_published" in c for c in cards.values())
    # LassoBench CMA-ES started at the origin (main text: "a published recipe, not verified; Appendix H")
    assert any("lassobench_pub:cmaes_full_published" in c for c in cards.values())
    # the three Linear BO variants: the port to the box, the shell port with calibrated bounds, the authors' box
    for card in ("linear_bo", "linear_bo_sphere_calib", "linear_bo_box8"):
        assert sum(card in c for c in cards.values()) == 1, card
    shell = next(r for r in rows.values() if ("linear_bo_sphere_calib", None) in r["_cards"])
    assert "calibrated" in shell["surrogate"] and "first run" in shell["support"]
    # COWBOYS: no printed table prints it (its Table 1 row was dropped); its card stays in the repository record
    assert "cowboys" in overview.OVERVIEW_OMITTED
    assert not any(c == "cowboys" for r in rows.values() for c, _ in r["_cards"])
    # review of 2026-10-24: the NAS-Bench-101 step size and space of GeoCMA, the Gaussian fit merged into the
    # random-search row, the split schedule of the submitted BAxUS run, the outputscale prior of the Xu et al. port
    g = rows["geocma"]
    assert "NAS: sd of cold starts" in g["acquisition"] and "NAS: whole space" in g["support"]
    rs = rows["random_search"]
    assert ("ellipsoid_rs", None) in rs["_cards"] and "N(m, C)" in rs["acquisition"] and "Gaussian" in rs["support"]
    assert {("ellipsoid_rs", "acquisition"), ("ellipsoid_rs", "support")} <= set(rs["_source"])
    # authors' decision of 2026-11-02: the stored LOL-BO, InvBO, TuRBO, BAxUS and CMA-ES runs have no overview row (their
    # cards stay in the repository record); the runs that Table 1 prints instead have one
    for gone in ("turbo", "baxus", "lolbo", "invbo", "cmaes_full"):
        assert gone not in rows, gone
    for card in ("turbo", "baxus", "lolbo", "invbo", "cma_es (full-dimensional)"):
        assert card in overview.OVERVIEW_OMITTED, card
    assert ("turbo_published", None) in rows["turbo_pub"]["_cards"]
    bx = rows["baxus_cr"]
    assert bx["group"] == "published" and bx["_cards"] == [("baxus_cr", None)]
    assert "BAxUS tutorial" in bx["implementation"] and ("baxus_cr", "trust region") in bx["_source"]
    cm = rows["cmaes_full_rd"]
    assert cm["group"] == "inrepo" and "population 5 or 20" in cm["acquisition"]
    assert ("cmaes_full_rd / cmaes_full_rd_pop5", None) in cm["_cards"]
    assert "no lengthscale/outputscale prior" in rows["x1_gp"]["surrogate"]


def test_written_gp_labels_are_prefixes_of_every_merged_label():
    for r in overview.load_overview():
        if r["group"] != "adapted" or not r["label"]:
            continue
        arms = [a for c, _ in r["_cards"] if not c.startswith("lassobench") for a in
                next(f for f in baselines.load_facts() if f["arm"] == c)["_run"] if a in labels.DESCRIPTIVE]
        assert arms
        for a in arms:
            assert labels.label(a).startswith(r["label"]), (r["row"], a)


@pytest.mark.parametrize("mutate,needle", [
    (lambda rows: rows[1].__setitem__(6, rows[1][6].replace("0.60", "0.65")), "number 0.65"),
    (lambda rows: rows[1].__setitem__(5, "Tanimoto kernel"), "keyword 'Tanimoto'"),
    (lambda rows: rows[1].__setitem__(4, "x" * (overview.MAX_CELL + 1)), "characters"),
    (lambda rows: rows[1].__setitem__(4, "initialised in this paper"), "forbidden text"),
    (lambda rows: rows.__delitem__(1), "neither covered nor omitted"),
    (lambda rows: rows[2].__setitem__(2, rows[2][2] + ";" + rows[1][2].split(";")[0]), "covered twice"),
    (lambda rows: rows[2].__setitem__(8, rows[2][8] + ";" + rows[1][8].split(";")[0]), "not a card of row"),
    (lambda rows: rows[1].__setitem__(7, "shell of radius {radius}"), "unknown token"),
    (lambda rows: rows[1].__setitem__(8, rows[1][8] + ";geowalk|nonexistent"), "not in appendixH_printed.csv"),
])
def test_overview_loader_refuses_a_broken_file(tmp_path, mutate, needle):
    rows = _rows()
    mutate(rows)
    with pytest.raises(overview.OverviewError, match=re.escape(needle)):
        overview.load_overview(_write(tmp_path, rows))


def test_split_card_must_cover_every_arm(tmp_path):
    rows = [r for r in _rows() if r[0] != "hvarfner_allobs"]
    with pytest.raises(overview.OverviewError, match="split by arm"):
        overview.load_overview(_write(tmp_path, rows))


def test_render_cell_sets_the_closed_notations_as_mathematics():
    assert overview.render_cell("shell of radius {r}; box [p01, p99]") == \
        "shell of radius $\\bar r$; box $[p_{01},p_{99}]$"
    assert overview.render_cell("Matern-5/2, start sqrt(D)/10; vendored MaternGaBO") == \
        "Mat\\'ern-5/2, start $\\sqrt{D}/10$; vendored MaternGaBO"
    assert overview.render_cell("a_b 100%") == "a\\_b 100\\%"
    assert overview.unsupported("argmax over 2,048 draws", "argmax over 2,048 fresh uniform points") == []
    assert overview.unsupported("argmax over 2,049 draws", "argmax over 2,048 fresh uniform points") == ["number 2049"]
    assert overview.unsupported("UCB and EI", "UpperConfidenceBound; EI instead of LogEI") == []
    assert overview.unsupported("qLogEI", "analytic LogExpectedImprovement") == ["keyword 'qLogEI'"]


# =====================================================================================================
# records: TeX -> Markdown, BASELINE_CARDS.md, RUN_STATUS.md
# =====================================================================================================
def test_tex_to_md_whitelist():
    md = records.tex_to_md("\\texttt{hvarfner\\_scaled\\_\\allowbreak{}gp}; Table~\\ref{tab:lassobench} $\\times$ 3"
                           "$^{\\ddagger}$ Mat\\'ern \\geowalk{} D\\textasciitilde{}=8 a|b \\emph{x}")
    assert md == "`hvarfner_scaled_gp`; the LassoBench table of the paper × 3‡ Matérn GeoWalk D\\~=8 a\\|b *x*"
    # the thousands separator (plain or escaped by the cards) is a comma; -- and --- are dashes outside code
    assert records.tex_to_md("4\\{,\\}420 steps; 2{,}048; seeds (42--81); a---b; \\texttt{--flag}") == \
        "4,420 steps; 2,048; seeds (42–81); a—b; `--flag`"
    for bad, msg in (("\\newcommand{\\x}{1}", "whitelist"), ("$\\alpha$", "MATH_MD"),
                     ("Table~\\ref{tab:nowhere}", "REF_NAMES"), ("{x}", "bare brace"), ("$x", "unclosed")):
        with pytest.raises(records.RecordError, match=msg):
            records.tex_to_md(bad)


#: the Markdown escapes tex_to_md writes: a backslash before a character of _MD_SPECIAL other than '$'
_MD_ESCAPE = re.compile(r"\\[" + re.escape(records._MD_SPECIAL.replace("$", "")) + "]")


def markdown_tex_residue(txt: str) -> list[str]:
    """TeX left in a written Markdown record: any of { } $ and --, a backslash that is not a Markdown escape of
    | * _ < > ~ [ ], or a tilde that is not escaped. (A ban of every backslash and tilde would forbid the escapes
    that keep Markdown from reading these characters as markup, e.g. linear\\_operator or \\[0.05, 0.50\\].)"""
    out = [f"{tok!r} at ...{txt[max(0, i - 30):i + 30]!r}..." for tok in ("{", "}", "$", "--")
           for i in [txt.find(tok)] if i >= 0]
    rest = _MD_ESCAPE.sub("", txt)
    for tok in ("\\", "~"):
        i = rest.find(tok)
        if i >= 0:
            out.append(f"unescaped {tok!r} at ...{rest[max(0, i - 30):i + 30]!r}...")
    return out


def test_markdown_residue_check_finds_tex():
    assert markdown_tex_residue("linear\\_operator, \\[0.05, 0.50\\], D\\~=8, 4,420, 42–81, `x`") == []
    for bad in ("4{,}420", "42--81", "$x$", "\\alpha", "D~=8", "a \\$ b"):
        assert markdown_tex_residue(bad), bad


def test_written_markdown_has_no_tex():
    """BASELINE_CARDS.md and RUN_STATUS.md: none of { } $ --, a backslash or a tilde only as a Markdown escape."""
    t = CardTable("baselines", "tab:baselines", printed=True)
    t.caption = "\\textbf{Cards.} Seeds 42--81; not checked (--); see Table~\\ref{tab:main}."
    t.group("Methods of this paper")
    t.card("\\textbf{GeoCMA D\\textasciitilde{}=8}", "cma_es_subspace_sd8")
    t.field("Seed count", "40 seeds (42--81) $\\times$ 11 tasks = 440 runs")
    t.field("Acquisition", "a uniform draw on the shell: 4\\{,\\}420 such steps; z \\textasciitilde{} N(m, C); "
                           "\\texttt{linear\\_operator}, best\\_f in [0.05, 0.50]")
    md = records.cards_markdown(t)
    assert "4,420 such steps" in md and "(42–81)" in md and "(–)" in md
    assert markdown_tex_residue(md) == [], markdown_tex_residue(md)
    s = Table("arm_status", "tab:arm_status", "\\textbf{Status.} Seeds 42--81.", "l", ["Method"] * 8)
    s.add(["GeoWalk", "this version", "4{,}400", "440", "0", "0$^{*}$", "0", "14481cb"])
    s.row_arms = ["geowalk"]
    md2 = records.arm_status_markdown(s)
    assert "| GeoWalk | `geowalk` | this version | 4,400 | 440 | 0 | 0\\* |" in md2 and "|-|-|" in md2
    assert markdown_tex_residue(md2) == [], markdown_tex_residue(md2)


def test_cards_and_status_markdown_carry_no_tex():
    t = CardTable("baselines", "tab:baselines", printed=True)
    t.caption = "\\textbf{Cards.} One card per method (Table~\\ref{tab:main})."
    t.group("Methods of this paper")
    t.card("\\textbf{\\geowalk{}}", "geowalk")
    t.field("Code name", "\\texttt{geowalk}")
    t.field("Kernel", "none (no surrogate)$^{\\ddagger}$")
    md = records.cards_markdown(t)
    assert "## Methods of this paper" in md and "### GeoWalk" in md and "- **Code name:** `geowalk`" in md
    assert not re.search(r"\\[A-Za-z]", md)
    s = Table("arm_status", "tab:arm_status", "\\textbf{Status.} Their code names are in the Appendix~H cards.",
              "l", ["Method"] * 8)
    s.add(["GeoWalk shell +-10\\%", "this version", "440", "440", "0", "0", "0", "14481cb"])
    s.row_arms = ["geowalk_shell10"]
    md2 = records.arm_status_markdown(s)
    assert "| GeoWalk shell +-10% | `geowalk_shell10` | this version | 440 |" in md2
    assert records.STATUS_CODE_NAMES[1] in md2
    s.row_arms = []
    with pytest.raises(records.RecordError, match="code names"):
        records.arm_status_markdown(s)


# =====================================================================================================
# tab_hyper: the table equals the code
# =====================================================================================================
def _import(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_settings_read_from_the_code_equal_the_imported_code():
    v = hyper.read_settings(REPO)
    sys.path.insert(0, REPO)
    try:
        from src.spherevo.solver import CMAESSubspaceSolver, GeoWalkSolver
    finally:
        sys.path.remove(REPO)
    gw = inspect.signature(GeoWalkSolver.__init__).parameters
    cm = inspect.signature(CMAESSubspaceSolver.__init__).parameters
    assert v["rho"] == gw["explore_ratio"].default == 0.64
    assert v["alpha_max"] == gw["angle_init"].default == 0.60
    assert v["alpha_min"] == gw["angle_final"].default == 0.15
    assert v["radius"] == gw["radius_estimator"].default == "mean"
    assert v["sigma0"] == cm["initial_sigma"].default == 0.5
    cell = _import(os.path.join(REPO, "scripts", "cr", "run_cell.py"), "run_cell_for_test")
    assert (v["N0"], v["T"], v["N"]) == (cell.RUN_DEFAULTS["n_cold_start"], cell.RUN_DEFAULTS["max_iter"], cell.BUDGET)
    assert v["subspace"] == cell.RUN_DEFAULTS["subspace_dim"] == 16
    assert v["N1"] == int(0.64 * 25) == 16 and v["N2"] == 9 and v["intervals"] == 8
    assert v["lambda"] == 5
    man = json.load(open(os.path.join(REPO, "scripts", "cr", "manifest.json")))
    assert v["tasks"] == man["tasks"] and len(v["tasks"]) == 11 and v["seeds"] == list(range(42, 82))
    assert v["timeout_s"] == 3600 and v["decode_cap"] == 80 and v["decode_seconds"] == 30 and v["zinc_pool"] == 10000


def _repo_copy(tmp_path):
    for rel in (hyper.SOLVER, hyper.RUNNER, hyper.RUN_CELL, hyper.MANIFEST, hyper.CODEC):
        dst = tmp_path / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(os.path.join(REPO, rel), dst)
    return tmp_path


def test_settings_follow_a_changed_default_and_refuse_a_factory_override(tmp_path):
    repo = _repo_copy(tmp_path)
    p = repo / hyper.SOLVER
    p.write_text(p.read_text().replace("explore_ratio: float = 0.64", "explore_ratio: float = 0.48", 1))
    v = hyper.read_settings(str(repo))
    assert v["rho"] == 0.48 and v["N1"] == 12 and v["N2"] == 13
    q = repo / hyper.RUNNER
    before = q.read_text()
    q.write_text(before.replace("            total_budget=n_iterations,\n        )",
                                "            total_budget=n_iterations, angle_init=0.3,\n        )", 1))
    assert q.read_text() != before
    with pytest.raises(hyper.HyperSourceError, match="geowalk factory"):
        hyper.read_settings(str(repo))


def test_hyper_table_prints_the_values_read():
    v = hyper.read_settings(REPO)
    ctx = types.SimpleNamespace(extra={"repo_root": REPO}, has=lambda a: False)
    t = hyper.tab_hyper(ctx)
    cells = {r[0]: r[1] for r in t.rows if isinstance(r, list)}
    assert cells["Exploration ratio $\\rho$"] == f"{v['rho']:.2f}" == "0.64"
    assert cells["Angle $\\alpha_{\\max}\\to\\alpha_{\\min}$"] == "$0.60\\to0.15$ rad"
    assert cells["Population size $\\lambda$"] == "5" and cells["Initial step size $\\sigma_0$"] == "0.5"
    assert cells["Subspace dimension $\\tilde D$"] == "16" and cells["Budget $N=N_0+T$"] == "30"
    assert "pycma not recorded" in cells["CMA-ES library and options"]
    assert t.label == "tab:hyper" and "Section~\\ref{sec:structured}" in t.caption
    assert not re.search(r"chosen|tuned|search", t.caption + " ".join(" ".join(r) for r in t.rows if isinstance(r, list)))


# =====================================================================================================
# tab_compute and the compute rule
# =====================================================================================================
def test_compute_rule_is_the_tool_rule(tmp_path):
    d = tmp_path / "blk"
    d.mkdir()
    recs = [{"elapsed_seconds": 10.0}, {"elapsed_s": 20}, {"wall_seconds": 5.0, "elapsed": 99},
            {"t_eval_s": [1.0, 2.0, "x"]}, {"nothing": 1}, [1, 2]]
    for i, r in enumerate(recs):
        (d / f"r{i}.json").write_text(json.dumps(r))
    (d / "broken.json").write_text("{")
    sc = runtime.scan(str(d))
    assert sc["files"] == 7 and sc["timed"] == 4
    assert sorted(sc["hosts"][None]["times"]) == [3.0, 5.0, 10.0, 20.0]
    tool = _import(os.path.join(ANALYSIS, "tools", "compute_summary.py"), "compute_summary_for_test")
    n, t = tool.times(str(d))
    assert n == 7 and sorted(t) == [3.0, 5.0, 10.0, 20.0]


def test_folder_host_names_the_files_without_a_host(tmp_path):
    d = tmp_path / "lasso"
    d.mkdir()
    (d / "env_provenance.txt").write_text("recorded 2026-10-12T13:18:51+02:00 on kajman.utia.cas.cz\nmore\n")
    (d / "a.json").write_text(json.dumps({"elapsed_s": 1.0}))
    (d / "b.json").write_text(json.dumps({"elapsed_s": 1.0, "provenance": {"hostname": "panda.utia.cas.cz"}}))
    sc = runtime.scan(str(d))
    assert set(sc["hosts"]) == {"kajman.utia.cas.cz", "panda.utia.cas.cz"}
    assert compute.machine_letter("kajman.utia.cas.cz") == "B" and compute.machine_letter("panda") == "A"
    assert compute.machine_letter("tulen.utia.cas.cz") == "C" and compute.machine_letter(None) == "not recorded"


class _Reg:
    def __init__(self):
        self.records = {}

    def add(self, name, value, text, **kw):
        assert name not in self.records
        self.records[name] = {"value": value, "text": text, **kw}


def _fake_ctx(tmp_path, roots, stored=True):
    reg = _Reg()
    arm = types.SimpleNamespace(absent=False, spec=types.SimpleNamespace(kind="stored", namespace=""))
    ctx = types.SimpleNamespace(ds=types.SimpleNamespace(config=types.SimpleNamespace(roots=roots),
                                                         arms={"turbo": arm} if stored else {}),
                                extra={"inputs_root": str(tmp_path / "inputs"),
                                       "lasso_cr_dir": str(tmp_path / "lasso_cr"),
                                       "lasso_pub_dir": str(tmp_path / "lasso_pub")},
                                cache={}, skipped=[])
    ctx.skip = lambda table, what, why: ctx.skipped.append((table, what, why))
    ctx.num = lambda name, value, text, **kw: reg.add(name, value, text, **kw)
    ctx.reg = reg
    return ctx


def _block(root, hours, host, n=2):
    os.makedirs(root, exist_ok=True)
    for i in range(n):
        rec = {"elapsed_seconds": hours * 3600 / n}
        if host:
            rec["provenance"] = {"hostname": host}
        with open(os.path.join(root, f"r{i}.json"), "w") as f:
            json.dump(rec, f)


def test_compute_table_totals_are_rounded_like_the_checklist(tmp_path):
    # the process-hours of the final run over the results bundle (2026-10-24): 644.30, 4,371.53, 4,184.84 and 1,140.48
    roots = {k: str(tmp_path / k) for k in ("CR", "CRPUB", "CRCTL", "CRCPU", "G")}
    _block(roots["CR"], 306.6283, "panda.utia.cas.cz")
    _block(roots["CRPUB"], 297.1161, "panda.utia.cas.cz")
    _block(roots["CRCTL"], 40.5590, "panda.utia.cas.cz")
    _block(roots["CRCPU"], 2.8752, "kajman.utia.cas.cz")
    _block(str(tmp_path / "inputs" / "verification_runs"), 4184.8353, "kajman.utia.cas.cz")
    _block(str(tmp_path / "lasso_pub"), 125.1823, "kajman.utia.cas.cz")
    _block(str(tmp_path / "lasso_cr"), 58.6383, None)         # no host in the files: env_provenance.txt names it
    (tmp_path / "lasso_cr" / "env_provenance.txt").write_text("recorded 2026-10-12T13:18:51+02:00 on kajman.utia.cas.cz\n")
    _block(roots["G"], 822.6, "kajman.utia.cas.cz")
    _block(os.path.join(roots["G"], "x"), 307.0, "tulen.utia.cas.cz")
    for d in ("nasbench", "chembed"):                 # stored runs under the inputs root that printed tables read
        _block(str(tmp_path / "inputs" / d), 1.0, "tulen.utia.cas.cz")
    _block(str(tmp_path / "inputs" / "lassobench"), 5.0, None)        # stored runs that no printed table reads
    _block(os.path.join(roots["G"], "y"), 10.8791, None)
    (tmp_path / "G" / "z.json").write_text(json.dumps({"no": "time"}))
    ctx = _fake_ctx(tmp_path, roots)
    t = compute.tab_compute(ctx)
    R = ctx.reg.records
    assert R["crComputeGpuServerHours"]["text"] == "644"
    assert R["crComputeCpuServerHours"]["text"] == "4{,}372"
    assert R["crComputeVerifyHours"]["text"] == "4{,}185"
    assert R["crComputeStoredHours"]["text"] == "1{,}140"
    rows = [r for r in t.rows if isinstance(r, list)]
    assert ["No time recorded", "--", "1", "0", "--", "--"] in rows
    assert any(r[0] == "Host not recorded" and r[4] == "10.9" for r in rows)
    assert any(r[0] == "Recorded on server C" and r[1] == "C" and r[4] == "307.0" for r in rows)
    assert t.label == "tab:compute" and "1 of these files record no time" in t.caption
    # machine B has the hardware of A; the runs of this version used its CPUs only (review of 2026-10-24)
    assert "B = the same CPUs and GPUs as A;" in t.caption and "with the GPUs hidden" in t.caption
    assert "6542Y; B" not in t.caption
    assert ("The runs of the submitted version are every result file of the stored folders that printed tables "
            "read: \\texttt{G} (GuacaMol, the rows without a benchmark name); \\texttt{chembed} (codec comparison, "
            "Table~\\ref{tab:crosscodec_means}); \\texttt{nasbench} (NAS-Bench-101, Table~\\ref{tab:crossdomain}).")\
        in t.caption
    assert ("In \\texttt{inputs/}, the stored LassoBench runs (\\texttt{lassobench}) are not counted, as no printed "
            "table reads them.") in t.caption
    # the stored folders that printed tables read: one row each; the GuacaMol total stays crComputeStoredHours, and
    # crComputeStoredPrintedHours adds the folders (the lassobench folder is not counted)
    assert ["Codec comparison: Chembed", "C", "2", "2", "1.0", "1{,}800.0"] in rows
    assert ["NAS-Bench-101", "C", "2", "2", "1.0", "1{,}800.0"] in rows
    assert next(r for r in rows if r[0] == "Runs of the submitted version, GuacaMol")[1:5] == \
        ["", "7", "6", "1{,}140.5"]
    assert next(r for r in rows if r[0] == "Runs of the submitted version, all folders above")[2:5] == \
        ["11", "10", "1{,}142.5"]
    assert R["crComputeStoredPrintedHours"]["text"] == "1{,}142"
    assert R["crComputeStoredPrintedHours"]["rule"].endswith(
        "printed tables read: root G (G), <inputs-root>/chembed, <inputs-root>/nasbench, rounded")
    assert R["crComputeStoredHours"]["rule"].startswith("process-hours of every result file (*.json) under root G")


def test_compute_table_skips_the_stored_runs_when_no_stored_arm_is_loaded(tmp_path):
    roots = {"CR": str(tmp_path / "CR"), "G": str(tmp_path / "G")}
    _block(roots["CR"], 1.0, "panda.utia.cas.cz")
    _block(roots["G"], 5.0, "kajman.utia.cas.cz")
    ctx = _fake_ctx(tmp_path, roots, stored=False)
    t = compute.tab_compute(ctx)
    assert "crComputeStoredHours" not in ctx.reg.records and "crComputeStoredPrintedHours" not in ctx.reg.records
    assert not any("submitted" in str(r) for r in t.rows)


# =====================================================================================================
# SOURCE.json of a generator output (the end-to-end test writes one)
# =====================================================================================================
def check_source_json(rdir: str) -> dict:
    src = json.load(open(os.path.join(rdir, records.SOURCE_JSON)))
    for name, sha in src["outputs_sha256"].items():
        assert hashlib.sha256(open(os.path.join(rdir, name), "rb").read()).hexdigest() == sha, name
    for rel, sha in src["inputs_sha256"].items():
        assert hashlib.sha256(open(os.path.join(REPO, rel), "rb").read()).hexdigest() == sha, rel
    return src
