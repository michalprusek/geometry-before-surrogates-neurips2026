"""GEN v5: camera-ready scope (RUN_RULES 2026-10-14), the published-recipe block (2026-10-12 (1), (4); 2026-10-13
revision 1; 2026-10-19), the five-contrast factorial of Table 2, the LassoBench published methods (design, provenance
of the result files, unread frozen variants, queue lists), the Papenmeier leukemia record check, and the
reader-facing text (no host names, no internal rules file name), running-text macros."""
import json
import os
import shutil
import subprocess

import numpy as np
import pytest

from crlib import baselines, labels, lasso_cr, published, scope, stats, stored, tables
from crlib.context import VARIANTS, Ctx
from crlib.latex import PREAMBLE_CHECK
from crlib.numbers import Registry, camel, text_macros
from v5_fixtures import PubSet, write_pub_design, write_verdicts

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYSIS = os.path.dirname(HERE)


# ---- scope -------------------------------------------------------------------------------------------------------
def test_scope_names_exist_and_cover_the_unprinted_content():
    import make_tables
    from crlib.config import load_config
    cfg = load_config(os.path.join(ANALYSIS, "sources_cr.json"), load_includes=False)
    builders = {b.__name__ for b in make_tables.TABLE_BUILDERS} | {"tab_posted_vs_printed", "tab_changes_vs_submitted"}
    assert scope.check_scope(set(cfg.arms), builders, set(VARIANTS), {f.__name__ for f in make_tables.NUMBER_FUNCTIONS}) == []
    # the control block, the GP radius arms and the stopped waves are never loaded in camera-ready mode
    ctl = [a for a, s in cfg.arms.items() if s.commit_root == "CRCTL"]
    assert ctl and set(ctl) <= set(scope.NOT_LOADED_ARMS)
    # control C4 reads the control block's directory through its own root and IS loaded: since RUN_RULES 2026-11-06 16:30
    # it is the shell x continuous cell of the printed factorial
    c4 = cfg.arms["hvarfner_scaled_gp_sphere_lbfgs"]
    assert c4.commit_root == "CRC4" and "hvarfner_scaled_gp_sphere_lbfgs" not in scope.NOT_LOADED_ARMS
    assert "hvarfner_scaled_gp_sphere_continuous" not in scope.NOT_LOADED_ARMS      # loaded, a repository record
    for a in ("hvarfner_scaled_gp_sphere05", "hvarfner_scaled_gp_box", "papenmeier_msr_published_norm_box",
              "random_box_faithful"):
        assert a in scope.NOT_LOADED_ARMS
    # the printed tables of the checklist (B.2) are not in the not-printed list
    for b in ("tab_main", "tab_boxsphere", "tab_lassobench", "tab_baselines", "tab_arm_status", "tab_geometry",
              "tab_published", "tab_verification"):
        assert b not in scope.NOT_PRINTED_TABLES
    # the frozen-noise published arms are not even listed in the sources
    assert "papenmeier_msr_reference" not in cfg.arms and "papenmeier_mle_scaled_reference" not in cfg.arms
    pub = [a for a, s in cfg.arms.items() if s.commit_root == "CRPUB"]
    assert sorted(pub) == sorted(published.GUACAMOL_ARMS)


def test_camera_ready_mode_refuses_a_variant_outside_the_scope(tmp_path, capsys):
    import make_tables
    rc = make_tables.main(["--outdir", str(tmp_path / "o"), "--expected-commit", "CR=" + "a" * 40,
                           "--variants", "R2_minmax", "--inputs-root", str(tmp_path)])
    assert rc == 2 and "not run in camera-ready mode" in capsys.readouterr().err


def test_geometry_keeps_zinc_only_in_camera_ready_mode(tmp_path):
    from crlib import geometry
    for col in ("default", "strip", "guacamol_train"):
        d = tmp_path / "g" / col
        d.mkdir(parents=True)
        (d / "geometry.json").write_text(json.dumps({"col": col}))
    for mode, want in (("cr", ["default", "strip"]), ("legacy", ["default", "strip", "guacamol_train"])):
        ctx = _ctx(tmp_path / mode, mode=mode)
        ctx.extra["geometry_root"] = str(tmp_path / "g")
        assert sorted(geometry.geometry_inputs(ctx)) == sorted(want)
    assert {"SumSigma", "ZNorm"} == set(scope.GEOMETRY_ROWS_DROPPED)


# ---- verdicts and names ------------------------------------------------------------------------------------------
def _ctx(tmp_path, verdicts_path=None, mode="cr"):
    out = tmp_path / f"out_{mode}"
    out.mkdir(parents=True, exist_ok=True)
    return Ctx(ds=None, eng=None, reg=Registry("x", "0" * 64, "default"), var=VARIANTS["default"], outdir=str(out),
               script_commit="x", mode=mode, extra={"inputs_root": str(tmp_path), "_shared_cache": {},
                                                    "verdicts_path": str(verdicts_path) if verdicts_path else None})


def test_verdict_file_is_pinned_and_decides_the_names(tmp_path, monkeypatch):
    p = tmp_path / "v" / "verdicts.json"
    write_verdicts(p)                                    # not pinned: the shipped sha256 is the real file's
    with pytest.raises(published.PublishedInputError, match="sha256"):
        published.load_verdicts(str(p))
    write_verdicts(p, monkeypatch=monkeypatch)
    V = published.load_verdicts(str(p))
    assert {r for r, v in V.items() if v["verified"]} == {"hvarfner", "msr", "mle_scaled", "xu", "doumont"}
    assert not V["turbo"]["verified"] and not V["baxus"]["verified"] and not V["cmaes"]["verified"]
    ctx = _ctx(tmp_path, p)
    assert published.name_or_label(ctx, "papenmeier_msr_reference_learned") == "MSR (Papenmeier et al.)"
    assert published.name_or_label(ctx, "turbo_published").endswith("(not verified)")
    assert "TuRBO" not in published.name_or_label(ctx, "turbo_published")
    assert published.name_or_label(ctx, "baxus_official", "lasso").endswith("(not verified)")
    # Hvarfner passes if either design passes; with both failing it keeps the descriptive label
    write_verdicts(p, {**{k: v for k, v in __import__("v5_fixtures").REAL_LIKE.items()},
                       ("hvarfner", 30): (False, False, 0)}, monkeypatch=monkeypatch)
    ctx2 = _ctx(tmp_path / "b", p)
    assert published.name_or_label(ctx2, "hvarfner_published") == labels.label("hvarfner_published") + " (not verified)"
    # a verdict for a frozen-noise variant (stopped unread on 2026-10-19) is refused
    write_verdicts(p, extra=[(("msr", 40), (True, True, 0))], monkeypatch=monkeypatch)
    with pytest.raises(published.PublishedInputError, match="stopped unread"):
        published.load_verdicts(str(p))
    # verdicts from another commit are refused, and so is a summarize pass at another commit or a dirty tree
    write_verdicts(p, commit="f" * 40, monkeypatch=monkeypatch)
    with pytest.raises(published.PublishedInputError):
        published.load_verdicts(str(p))
    for kw in ({"summarizer_commit": "e" * 40}, {"summarizer_dirty": True}):
        write_verdicts(p, monkeypatch=monkeypatch, **kw)
        with pytest.raises(published.PublishedInputError, match="summarize ran at"):
            published.load_verdicts(str(p))


def test_reading_of_rule_4():
    assert published.reading(-0.02, 0.01, 1, 10, 11, "X", "R")[0] == "better"
    assert published.reading(0.02, 0.01, 10, 1, 11, "X", "R")[0] == "worse"
    # RUN_RULES 2026-10-22 (wording only): the arm with the better mean is the subject, then its tasks, then the
    # corrected p and "not significant after correction"; the words "not worse" are never printed
    for d, p in ((0.02, 0.2), (-0.02, 0.2), (0.0, float("nan"))):
        word, sent = published.reading(d, p, 6, 5, 11, "X", "R")
        assert word == "not worse" and "not worse" not in sent and "not significant after correction" in sent
    assert published.reading(0.02, 0.2, 6, 5, 11, "X", "R")[1].startswith(
        "X has a higher mean than R and is higher on 6 of the 11 tasks; the difference is not significant")
    assert published.reading(-0.02, 0.2, 6, 5, 11, "X", "R")[1].startswith(
        "R has a higher mean than X and is higher on 5 of the 11 tasks; the difference is not significant")
    assert published.reading(0.0, float("nan"), 6, 5, 11, "X", "R")[1].startswith("X and R have the same mean")
    assert published.VERDICT_TEXT["not worse"] == "no significant difference"
    w, s = published.reading(-0.02, 0.01, 1, 10, 11, "X", "R")
    assert s.startswith("R is ahead of X (higher on 10 of the 11")
    # RUN_RULES 2026-10-19 second set (a): the third case has the reference as subject and names the recipe
    w, s = published.reading(0.02, 0.01, 10, 1, 11, "X", "Recipe R")
    assert w == "worse" and s.startswith("X is ahead of Recipe R (higher on 10 of the 11 tasks")


def test_reading_on_a_loss_says_lower_loss():
    """Review B1: on LassoBench the direction of improvement is a lower loss; the sentence must not say 'higher'."""
    for d, p, word in ((-0.02, 0.01, "better"), (0.02, 0.01, "worse"), (0.02, 0.3, "not worse")):
        w, s = published.reading(d, p, 5, 1, 6, "\\geowalk{}", "R", ahead="lower loss")
        assert w == word and "higher" not in s.replace("$p_{\\mathrm{BH}}", "") and "lower" in s, s
    w, s = published.reading(0.02, 0.3, 5, 1, 6, "\\geowalk{}", "R", ahead="lower loss")
    assert s.startswith("\\geowalk{} has a lower mean loss than R and is lower on 5 of the 6 tasks")
    w, s = published.reading(-0.02, 0.3, 1, 5, 6, "\\geowalk{}", "R", ahead="lower loss")
    assert s.startswith("R has a lower mean loss than \\geowalk{} and is lower on 5 of the 6 tasks")


def test_geowalk_vs_shell_gp_reading_uses_the_task_level_test():
    class P:
        pass
    pr = P()
    pr.delta_taskmean, pr.tasks_won, pr.tasks_lost, pr.n_tasks, pr.p = 0.01, 8, 3, 11, 1e-6
    pr.p_task = 0.2
    s = tables.geowalk_vs_shell_gp_reading(pr)
    assert s.startswith("no detected difference at $N=30$") and "equivalent" not in s
    pr.p_task, pr.p = 0.01, 0.6
    assert tables.geowalk_vs_shell_gp_reading(pr).startswith("\\geowalk{} is ahead (higher on 8 of the 11")
    pr.delta_taskmean = -0.01
    assert tables.geowalk_vs_shell_gp_reading(pr).startswith("the GP on the shell is ahead (higher on 3 of the 11")


def test_smaller_margin_rule_of_clause_c():
    d, ents = tables.submitted_t2_delta("Papenmeier")
    assert str(d) == "0.0355" and [e["id"] for e in ents] == ["S-t2-crBoxPapenmeierSphereMean", "S-t2-crBoxPapenmeierBoxMean"]
    assert [e["source"] for e in ents] == ["main.tex:212", "main.tex:213"] and ents[0]["printed"] == "0.4149"
    assert tables.submitted_t2_delta("VanillaBO") is None             # the rule names the Papenmeier row only
    assert tables.smaller_margin(0.02, 0.01, d)                       # leads, significant, smaller
    assert not tables.smaller_margin(0.02, 0.06, d)                   # (i) not significant after BH
    assert not tables.smaller_margin(-0.02, 0.01, d)                  # (i) the box leads
    assert not tables.smaller_margin(0.04, 0.01, d)                   # (ii) not smaller
    assert not tables.smaller_margin(0.0355, 0.01, d)                 # (ii) equal is not smaller
    assert not tables.smaller_margin(0.02, float("nan"), d)
    # second set (d): compared at the printed precision; a tie there is not smaller
    assert not tables.smaller_margin(0.03549, 0.01, d)                # prints +0.0355: a tie, not printed
    assert tables.smaller_margin(0.03544, 0.01, d)                    # prints +0.0354: smaller, printed
    assert not tables.smaller_margin(0.035501, 0.01, d)


def test_changes_table_is_a_record_only():
    assert "tab_changes_vs_submitted" in scope.RECORD_ONLY_TABLES
    assert "tab_changes_vs_submitted" not in scope.NOT_PRINTED_TABLES
    # RUN_RULES 2026-10-24: the cards and the run status are repository records; the overview is printed and is the
    # last builder (it summarizes the cards that tab_baselines produced)
    import make_tables
    for b in ("tab_baselines", "tab_arm_status"):
        assert b in scope.RECORD_ONLY_TABLES and b not in scope.NOT_PRINTED_TABLES
    names = [b.__name__ for b in make_tables.TABLE_BUILDERS]
    assert names[-1] == "tab_baselines_overview" and names.index("tab_baselines") < names.index("tab_baselines_overview")
    assert {"tab_hyper", "tab_compute"} <= set(names) and not {"tab_hyper", "tab_compute"} & set(scope.NOT_PRINTED_TABLES)


def test_factorial_reading_says_significant_only_after_bh():
    class P:
        pass
    pr = P()
    pr.delta_taskmean, pr.tasks_won, pr.tasks_lost, pr.n_tasks = 0.01, 9, 2, 11
    pr.p_task_bh = 0.2
    lead, s = tables.factorial_reading(pr, "pos", "neg")
    assert lead == "pos" and s == "pos on 9 of the 11 tasks" and "significant" not in s
    pr.p_task_bh = 0.01
    assert "significant after Benjamini--Hochberg over the five" in tables.factorial_reading(pr, "pos", "neg")[1]
    pr.delta_taskmean = -0.01
    assert tables.factorial_reading(pr, "pos", "neg")[1].startswith("neg on 2 of the 11 tasks")
    assert [k for k, *_ in tables.FACTORIAL_FIVE] == ["SupportCand", "SupportCont", "AcqShell", "AcqBox", "Interaction"]


def test_descriptive_labels_never_carry_the_authors_names():
    names = ("Hvarfner", "Papenmeier", "Xu", "Doumont", "Vanilla BO", "Linear BO", "TuRBO", "BAxUS", "LOL-BO", "InvBO",
             "SAASBO", "REMBO", "HeSBO", "GaBO", "COWBOYS")
    for arm, (lab, plain) in labels.DESCRIPTIVE.items():
        assert not any(n in lab or n in plain for n in names), arm
    for arm, (lab, plain) in labels.LASSO_DESCRIPTIVE.items():
        assert not any(n in lab or n in plain for n in names), arm
    rows = [lab for _, lab in stored.LASSO_METHODS]
    assert not any(n in r for r in rows for n in names)


# ---- LassoBench published methods --------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def pub_design(tmp_path_factory):
    root = tmp_path_factory.mktemp("pub") / "lassobench_pub"
    write_pub_design(root, loss=lambda m, t, s: 1.0 + 0.001 * s + 0.01 * len(m),
                     timeout=[("papenmeier_published", "leukemia", 50)])
    return str(root)


def test_published_design_follows_the_pilots_and_is_finished(pub_design):
    L = lasso_cr.load_rerun(pub_design, design=lasso_cr.PUB, require_finished=True)
    s = L["summary"]
    assert s["finished"] and s["jobs_expected"] == 2010 + 7 * 39 and s["timeout"] == 1 + 3      # + 3 pilots over the hour
    cov = L["coverage"]
    assert cov[("papenmeier_published_learned", "leukemia")]["pilot"] and cov[("hvarfner_published", "leukemia")]["expected"] == 40
    assert L["leukemia"]["papenmeier_published_learned"]["status"] == "not_completed"
    assert L["leukemia"]["xu_published"]["status"] == "not_applicable"
    # the frozen-noise variants are checked (record, sha256) but never parsed: their files are not JSON here
    assert cov[("papenmeier_published", "dna")]["unread"] == lasso_cr.SEEDS
    assert "papenmeier_published" not in L["cells"] and s["unread"] == 2 * 200 + 39 + 40
    assert cov[("xu_published", "synt_hard")]["not_applicable"] == lasso_cr.SEEDS


def test_published_result_provenance_and_queue_lists_are_checked(tmp_path, pub_design):
    import shutil
    bad = tmp_path / "bad_prov"
    shutil.copytree(pub_design, bad)
    shutil.copytree(pub_design + "_queue", str(bad) + "_queue")
    P = PubSet(bad)
    os.remove(os.path.join(P.g, "results", "hvarfner_published_dna_s42.json"))
    lines = [ln for ln in open(os.path.join(P.g, "manifest.jsonl")) if '"hvarfner_published_dna_s42"' not in ln]
    open(os.path.join(P.g, "manifest.jsonl"), "w").writelines(lines)
    P.job("hvarfner_published_dna_s42", "hvarfner_published", "dna", 42, 1.0, prov={"git_commit": "", "git_dirty": None})
    with pytest.raises(lasso_cr.LassoLoadError, match="provenance.git_commit"):
        lasso_cr.load_rerun(str(bad), design=lasso_cr.PUB)
    q = tmp_path / "bad_queue"
    shutil.copytree(pub_design, q)
    shutil.copytree(pub_design + "_queue", str(q) + "_queue")
    with open(os.path.join(str(q) + "_queue", "queue", "jobs_phase1.txt"), "a") as f:
        f.write("bash /c/run_lasso.sh pub_methods extra_job results /r/x.py hvarfner_published dna 99 3600\n")
    with pytest.raises(lasso_cr.LassoLoadError, match="not in the expected design"):
        lasso_cr.load_rerun(str(q), design=lasso_cr.PUB)


def test_published_lassobench_rows_and_family(tmp_path, pub_design, monkeypatch):
    from test_lasso_cr import write_original_design
    R = write_original_design(tmp_path / "lassobench_cr")
    R.papenmeier_leukemia(range(43, 82))
    p = tmp_path / "v" / "verdicts.json"
    write_verdicts(p, monkeypatch=monkeypatch)
    ctx = _ctx(tmp_path, p)
    ctx.extra.update(lasso_cr_dir=R.root, lasso_pub_dir=pub_design)
    t = stored.tab_lassobench(ctx)
    rows = {r[0]: r for r in t.rows if isinstance(r, list)}
    assert rows["MSR (Papenmeier et al.)"][-1] == "n.c." and rows["Standard GP (Xu et al.)"][-2] == "n.a."
    assert rows["Standard GP (Xu et al.)"][-1] == "n.a."
    assert stored.lasso_method_tex("turbo") not in rows and stored.lasso_method_tex("cmaes_full") not in rows
    assert any(k.endswith("$^{\\circ}$") for k in rows)   # RUN_RULES 2026-11-02 (3)
    # second set (c): the LassoBench row labels follow the print rule too (review B3)
    from v5_fixtures import label_problems
    names = [published.RECIPES[r]["name"] for r in ("hvarfner", "msr", "mle_scaled", "xu", "doumont")]
    assert label_problems([("tab_lassobench", k) for k in rows] + [("block", r[0]) for b in t.blocks for r in b[3]],
                          names) == []
    P = stored.lasso_pub_analysis(ctx)
    assert len(P["printed"]) == 8 and "papenmeier_published" not in P["printed"]
    ps = [P["fam"][m]["p_task"] for m in P["printed"]]
    assert [P["fam"][m]["p_task_bh"] for m in P["printed"]] == pytest.approx(stats.bh(ps), nan_ok=True)
    rec = ctx.reg.records
    assert rec["crLassoPubHvarfnerPublishedVsGeowalkVerdict"]["value"] in ("better", "worse", "not worse")
    # review B1: every LassoBench rule-(4) sentence speaks of the lower loss, with the tasks GeoWalk is lower on
    for m in P["printed"]:
        k = f"crLassoPub{camel(m)}VsGeowalk"
        sent, f = rec[k + "Sentence"]["value"], P["fam"][m]
        assert "higher" not in sent and "lower" in sent and "not worse" not in sent, sent
        verdict = rec[k + "Verdict"]
        assert verdict["text"] == published.VERDICT_TEXT[verdict["value"]] and "not worse" not in verdict["text"]
        if rec[k + "Verdict"]["value"] == "not worse":
            lead_gw = f["delta"] < 0          # GeoWalk minus method: negative = GeoWalk has the lower loss
            k_tasks = f["gw_lower"] if lead_gw else f["mth_lower"]
            assert f"and is lower on {k_tasks} of the {f['n_tasks']} tasks" in sent, sent
            assert sent.startswith("\\geowalk{} has a lower mean loss") == lead_gw, sent
    assert "lassobench_pub_coverage.md" in " ".join(ctx.written)


def test_make_tables_refuses_an_unfinished_published_set(tmp_path, pub_design, monkeypatch):
    import shutil
    import make_tables
    from test_lasso_cr import write_original_design
    R = write_original_design(tmp_path / "lassobench_cr")
    R.papenmeier_leukemia(range(43, 82))
    part = tmp_path / "part"
    shutil.copytree(pub_design, part)
    shutil.copytree(pub_design + "_queue", str(part) + "_queue")
    lines = [ln for ln in open(part / "pub_methods" / "manifest.jsonl") if '"xu_published_dna_s81"' not in ln]
    open(part / "pub_methods" / "manifest.jsonl", "w").writelines(lines)
    os.remove(part / "pub_methods" / "results" / "xu_published_dna_s81.json")
    os.remove(part / "pub_methods" / "status" / "xu_published_dna_s81.rc")

    def stop(*a, **k):
        raise make_tables.ConfigError("stopped by the test after the input checks")
    monkeypatch.setattr(make_tables, "load_config", stop)
    p = tmp_path / "v" / "verdicts.json"
    write_verdicts(p, monkeypatch=monkeypatch)
    base = ["--expected-commit", "CR=" + "a" * 40, "--inputs-root", str(tmp_path), "--lasso-cr-root", R.root,
            "--verdicts", str(p), "--lasso-require-finished", "--validation-run"]
    rc = make_tables.main(base + ["--outdir", str(tmp_path / "o1"), "--lasso-pub-root", str(part)])
    failed = (tmp_path / "o1" / "LOADER_FAILED.txt").read_text()
    assert rc == 2 and failed.startswith("LASSOBENCH PUBLISHED-METHOD SET NOT USABLE FOR A FINAL RUN")
    assert "xu_published dna: no job record yet: seeds 81" in failed
    rc = make_tables.main(base + ["--outdir", str(tmp_path / "o2"), "--lasso-pub-root", pub_design])
    assert rc == 2 and not (tmp_path / "o2" / "LOADER_FAILED.txt").exists()          # stopped by the stub: accepted
    write_verdicts(p, extra=[(("other", 1), (True, True, 0))])                       # another file: not pinned
    rc = make_tables.main(base + ["--outdir", str(tmp_path / "o3"), "--lasso-pub-root", pub_design])
    assert rc == 2 and "VERIFICATION VERDICTS NOT USABLE" in (tmp_path / "o3" / "LOADER_FAILED.txt").read_text()


def test_papenmeier_leukemia_records_must_match_the_pilot(tmp_path):
    from test_lasso_cr import write_original_design
    R = write_original_design(tmp_path / "lassobench_cr")
    R.papenmeier_leukemia(range(43, 82))
    lasso_cr.load_rerun(R.root, require_finished=True)          # identical records: accepted
    m = os.path.join(R.root, "full_rerun", "manifest.jsonl")
    recs = [json.loads(ln) for ln in open(m)]
    for r in recs:
        if r.get("label") == "papenmeier_leukemia_s60":
            r["timeout_s"] = "7200"
    open(m, "w").write("".join(json.dumps(r) + "\n" for r in recs))
    with pytest.raises(lasso_cr.LassoLoadError, match="timeout_s differs from the pilot's"):
        lasso_cr.load_rerun(R.root)


# ---- reader-facing text ------------------------------------------------------------------------------------------
def test_cards_name_machines_by_hardware_and_never_print_the_rules_file_name():
    txt = baselines.reader_facing(baselines.neutral_hosts(baselines.tex_escape(
        "camera-ready runs on panda.utia.cas.cz and kajman (RUN_RULES R-2); tulen; pandas 2.2")))
    assert "panda" not in txt.replace("pandas", "") and "kajman" not in txt and "tulen" not in txt
    assert "server A" in txt and "server B" in txt and "server C" in txt and "pandas 2.2" in txt
    assert "RUN\\_RULES" not in txt and "the run rules R-2" in txt
    rows = baselines.load_facts()
    assert len(rows) == baselines.N_AUDITED_ROWS + 13 + 14 + 2 + 1     # + 2: the GuacaMol runs of 2026-11-02 (CRRD)
    groups = {r["table group"] for r in rows}
    assert {"Published recipes (released-code settings)", "LassoBench published methods (released-code settings)"} <= groups
    new = [r for r in rows if r["row origin"].startswith("GEN v5")]
    assert len(new) == 14 and all(r["needs proofreading"] == "all" for r in new)
    run = {a for r in new for a in r["_run"]}
    assert run == set(published.GUACAMOL_ARMS)


def test_generated_text_does_not_cite_the_internal_rules_file():
    import glob
    for p in glob.glob(os.path.join(ANALYSIS, "crlib", "*.py")):
        # every string printed on the page uses the reader-facing phrase (latex.rules), never the escaped file name;
        # baselines.reader_facing is the one place that rewrites it (in facts-table text)
        src = open(p).read()
        if p.endswith("baselines.py"):
            src = src.replace('latex.replace("RUN\\\\_RULES", "the run rules")', "")
        assert "RUN\\\\_RULES" not in src, p


# ---- Table 2: the "smaller margin" sentence through the real builder ---------------------------------------------
T2_ARMS = ("hvarfner_scaled_gp", "hvarfner_scaled_gp_box_faithful", "papenmeier_msr", "papenmeier_msr_published",
           "xu_standard_gp", "xu_standard_gp_box_faithful")


def _t2_ctx(tmp_path, write_config, pap_shell, pap_box, others=None):
    """Six recipe arms on the 11 tasks x seeds 42-43; best@30 = level + a task offset (+ a tiny seed term).
    ``pap_shell`` / ``pap_box``: level, or a function task index -> level."""
    from conftest import arm_cfg, ours_record
    from crlib import TASKS_T11
    from crlib.config import load_config
    from crlib.engine import Engine
    from crlib.loader import load_all
    tasks = list(TASKS_T11)
    lev = {"hvarfner_scaled_gp": 0.42, "hvarfner_scaled_gp_box_faithful": 0.39, "xu_standard_gp": 0.41,
           "xu_standard_gp_box_faithful": 0.38, "papenmeier_msr": pap_shell, "papenmeier_msr_published": pap_box}
    lev.update(others or {})
    for arm in T2_ARMS:
        d = tmp_path / arm
        d.mkdir(parents=True, exist_ok=True)
        for i, t in enumerate(tasks):
            for s in (42, 43):
                base = lev[arm](i) if callable(lev[arm]) else lev[arm]
                top = base + 0.01 * i + 0.0001 * (s - 42)
                rec = ours_record(arm, t, s, scores=[0.1] * 29 + [top])
                rec["provenance"]["cr_arm"] = arm
                (d / f"ours_{arm}_{TASKS_T11[t]}_s{s}_128d.json").write_text(json.dumps(rec))
    cfgp = write_config({a: arm_cfg(a, cr_arm=a) for a in T2_ARMS}, tasks=tuple(tasks), seeds="42-43")
    ds = load_all(load_config(cfgp), "abc1234", log=lambda m: None)
    return Ctx(ds=ds, eng=Engine(ds), reg=Registry("x", "0" * 64, "default"), var=VARIANTS["default"],
               outdir=str(tmp_path), script_commit="x", mode="cr", extra={"_shared_cache": {}})


@pytest.mark.parametrize("shell, box, holds", [
    (0.42, 0.40, True),                                   # leads on every task by 0.02 < 0.0355: printed
    (0.42, 0.37, False),                                  # leads by 0.05: not smaller than the submitted margin
    (0.40, 0.42, False),                                  # the box leads
    (lambda i: 0.42 if i % 2 == 0 else 0.40, 0.41, False),  # mixed tasks: no significant lead after BH
    # leads on 9 of 11 tasks: exact task-level p < 0.05 uncorrected, but not after BH within the three rows (the other
    # two rows show no lead), so the sentence is NOT printed
    (lambda i: 0.40 + (-1 if i in (2, 4) else 1) * 0.002 * (i + 1), 0.40, "uncorrected-only"),
])
def test_table2_smaller_margin_sentence(tmp_path, write_config, shell, box, holds):
    others = None
    if holds == "uncorrected-only":
        flat = {a: (lambda i: 0.39 + (0.005 if i % 2 == 0 else -0.005)) for a in ("hvarfner_scaled_gp", "xu_standard_gp")}
        others = {**flat, "hvarfner_scaled_gp_box_faithful": 0.39, "xu_standard_gp_box_faithful": 0.39}
    ctx = _t2_ctx(tmp_path, write_config, shell, box, others)
    if holds == "uncorrected-only":
        rec = ctx.reg.records
        tables.tab_boxsphere(ctx)
        assert rec["crBoxPapenmeierPTask"]["value"] < 0.05 <= rec["crBoxPapenmeierPTaskBH"]["value"]
        assert rec["crBoxPapenmeierDelta"]["value"] > 0 and rec["crBoxPapenmeierSmallerMarginHolds"]["value"] == "no"
        assert rec["crBoxPapenmeierSmallerMargin"]["text"] == ""
        return
    t = tables.tab_boxsphere(ctx)
    assert t is not None
    rec = ctx.reg.records
    assert rec["crBoxPapenmeierSubmittedDelta"]["value"] == pytest.approx(0.0355)
    assert rec["crBoxPapenmeierSmallerMarginHolds"]["value"] == ("yes" if holds else "no")
    assert rec["crBoxPapenmeierSmallerMargin"]["text"] == (tables.SMALLER_MARGIN_SENTENCE if holds else "")
    q = stats.bh([rec[f"crBox{k}PTask"]["value"] for k in ("VanillaBO", "Papenmeier", "XuStandardGP")])
    assert rec["crBoxPapenmeierPTaskBH"]["value"] == pytest.approx(q[1])
    assert "crBoxVanillaBOSmallerMargin" not in rec            # the rule names the Papenmeier row only


# ---- running-text macros ---------------------------------------------------------------------------------------
_FS = {"count": 1, "list_sha256": "0" * 64, "content_sha256": "0" * 64}


def _text_registry(sentence: str) -> Registry:
    reg = Registry("x", "0" * 64, "default")
    reg.add("crBoxPapenmeierSmallerMargin", sentence, sentence, kind="text", table="boxsphere", fileset=_FS)
    reg.add("crBoxPapenmeierSmallerMarginHolds", "yes" if sentence else "no", "yes" if sentence else "no",
            kind="text", table="boxsphere", fileset=_FS)
    reg.add("crBoxPapenmeierDelta", 0.02, "+0.0200", kind="delta", table="boxsphere", fileset=_FS)
    reg.add("crBoxPapenmeierPTask", float("nan"), "", kind="p_task", table="boxsphere", fileset=_FS)
    return reg


def test_running_text_macros_are_written_in_text_mode(tmp_path):
    for sentence in (tables.SMALLER_MARGIN_SENTENCE, ""):
        path = tmp_path / "numbers.tex"
        _text_registry(sentence).write_tex(str(path), "test")
        tex = path.read_text()
        # a sentence is running text: no \ensuremath, no \text box; the empty sentence stays empty (not "--")
        assert "\\newcommand{\\crBoxPapenmeierSmallerMargin}{" + sentence + "}\n" in tex
        assert "\\newcommand{\\crBoxPapenmeierSmallerMarginHolds}{" + ("yes" if sentence else "no") + "}\n" in tex
        # numbers keep math mode, a missing number stays "--"
        assert "\\newcommand{\\crBoxPapenmeierDelta}{\\ensuremath{+0.0200}}\n" in tex
        assert "\\newcommand{\\crBoxPapenmeierPTask}{\\ensuremath{--}}\n" in tex
        assert text_macros(str(path)) == ["crBoxPapenmeierSmallerMargin", "crBoxPapenmeierSmallerMarginHolds"]


@pytest.mark.skipif(not shutil.which("pdflatex"), reason="pdflatex not available")
@pytest.mark.parametrize("sentence", [tables.SMALLER_MARGIN_SENTENCE, ""])
def test_smaller_margin_sentence_breaks_in_a_paragraph(tmp_path, sentence):
    """At the NeurIPS text width the clause-(c) sentence must break like ordinary text (as a \\text box it ran
    183pt into the margin); the empty case must leave nothing behind."""
    _text_registry(sentence).write_tex(str(tmp_path / "numbers.tex"), "test")
    (tmp_path / "check.tex").write_text(
        PREAMBLE_CHECK + "\\input{numbers.tex}\n"
        "For the corrected Papenmeier recipe of Table 2 the shell arm is ahead of the box arm on most tasks; "
        "\\crBoxPapenmeierSmallerMargin\\ (Holds: \\crBoxPapenmeierSmallerMarginHolds).\n\\end{document}\n")
    r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "check.tex"], cwd=tmp_path,
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout[-2000:]
    log = (tmp_path / "check.log").read_text(errors="replace")
    assert "Overfull \\hbox" not in log


# ---- LassoBench baseline family: paired on the seeds both arms completed (review B2) --------------------------
def test_lassobench_family_is_paired_on_the_seeds_both_completed(tmp_path, monkeypatch):
    from test_lasso_cr import write_original_design
    R = write_original_design(tmp_path / "lassobench_cr")
    R.papenmeier_leukemia(range(43, 82))
    hard = [("cmaes_full_published", "dna", s) for s in range(62, 82)]        # the 20 hard seeds time out

    def loss(m, t, s):
        return 0.5 + 0.02 * (s - 42) if (m, t) == ("cmaes_full_published", "dna") else 1.0 + 0.001 * s
    pub = tmp_path / "lassobench_pub"
    write_pub_design(pub, loss=loss, timeout=hard)
    p = tmp_path / "v" / "verdicts.json"
    write_verdicts(p, monkeypatch=monkeypatch)
    ctx = _ctx(tmp_path, p)
    ctx.extra.update(lasso_cr_dir=R.root, lasso_pub_dir=str(pub))
    t = stored.tab_lassobench(ctx)
    P = stored.lasso_pub_analysis(ctx)
    Rr = stored.lasso_rerun_analysis(ctx)
    g, norm = Rr["loaded"]["cells"]["geowalk"]["dna"], Rr["norm"]["dna"]
    f = P["fam"]["cmaes_full_published"]
    i = f["tasks"].index("dna")
    paired = np.mean([g[s][0] - loss("cmaes_full_published", "dna", s) for s in range(42, 62)]) / norm
    own = (np.mean([v for v, _ in g.values()]) - np.mean([loss("cmaes_full_published", "dna", s)
                                                          for s in range(42, 62)])) / norm
    assert f["diff"][i] == pytest.approx(paired) and abs(paired - own) > 1e-3
    assert f["pairs"]["dna"] == 20 and f["n_pairs"] == sum(f["pairs"].values())
    rec = ctx.reg.records
    assert rec["crLassoPubCmaesFullPublishedVsGeowalkN"]["value"] == f["n_pairs"]
    # the matched n is printed in the family block; the per-dimension cell stays the mean over its own 20 seeds
    blk = t.blocks[-1]
    name = published.name_or_label(ctx, "cmaes_full_published", "lasso")
    # second pass of 2026-10-29: the methods without a GuacaMol run are marked as LassoBench only
    row = [r for r in blk[3] if r[0].replace("$^{\\S}$", "") == name.replace(" (not verified)", "$^{\\circ}$")][0]
    assert row[3] == str(f["n_pairs"]) and row[0].endswith("$^{\\S}$")
    assert rec["crLassoPubCmaesFullPublishedDOneEightZeroN"]["value"] == 20


# ---- printed labels everywhere (RUN_RULES 2026-10-19 second set (b), (c); review B3, S5) ------------------------
def test_every_loaded_arm_has_a_printed_label_without_code_or_author_names(tmp_path, monkeypatch):
    """Card titles and arm-status rows use published.printed_label: every arm the camera-ready run loads (stored
    budget and codec arms included) resolves to a label that carries an author's or method's name only inside a
    verified recipe's printed name, and never a code name."""
    from types import SimpleNamespace
    from crlib.config import load_config
    from v5_fixtures import label_problems
    cfg = load_config(os.path.join(ANALYSIS, "sources_cr.json"), load_includes=False)
    p = tmp_path / "v" / "verdicts.json"
    write_verdicts(p, monkeypatch=monkeypatch)
    ctx = _ctx(tmp_path, p)
    ctx.ds = SimpleNamespace(arms={n: SimpleNamespace(spec=s) for n, s in cfg.arms.items()})
    monkeypatch.setattr(published, "hvarfner_arm", lambda c: "hvarfner_published")
    arms = [a for a in cfg.arms if a not in scope.NOT_LOADED_ARMS and not a.endswith("__s82")]
    labs = [("printed_label", published.printed_label(ctx, a)) for a in arms]
    names = [published.RECIPES[r]["name"] for r in ("hvarfner", "msr", "mle_scaled", "xu", "doumont")]
    assert label_problems(labs, names) == []
    assert len({lab for _, lab in labs}) == len(labs)                 # distinct: the arm-status rows stay apart
    assert published.printed_label(ctx, "hvarfner_published") == "Vanilla BO (Hvarfner et al.)"
    assert "Hvarfner" not in published.printed_label(ctx, "hvarfner_published_allobs")
    others = [("card", published.other_row_label(ctx, r["arm"])) for r in baselines.load_facts() if not r["_run"]]
    assert len(others) == 21 and label_problems(others, names) == []
    # clause (b): CMA-ES keeps its algorithm name with its settings and the mark; TuRBO and BAxUS stay descriptive
    assert published.name_or_label(ctx, "cmaes_full_published", "lasso") == \
        "CMA-ES, popsize 20, $\\sigma_0=0.1$, $x_0=0$ (not verified)"
    for m in ("turbo_published", "baxus_official"):
        lab = published.name_or_label(ctx, m, "lasso")
        assert lab.endswith("(not verified)") and "TuRBO" not in lab and "BAxUS" not in lab


def test_rerun_task_level_test_is_paired_on_the_seeds_both_completed(tmp_path):
    """Review B2 (consistency): the rerun's GeoWalk-vs-GeoCMA task-level test uses the same pairing as the family."""
    from test_lasso_cr import write_original_design
    R = write_original_design(tmp_path / "lassobench_cr", nonfinite_seed_81=("geowalk",))
    L = lasso_cr.load_rerun(R.root)
    S = stored.lasso_stats(L)
    c, norm = L["cells"], S["norm"]
    tasks = [t for t, _ in stored.LASSO_TASKS if t in norm]
    tdm = []
    for t in tasks:
        both = sorted(set(c["geowalk"][t]) & set(c["cmaes_sub"][t]))
        tdm.append(np.mean([c["geowalk"][t][s][0] - c["cmaes_sub"][t][s][0] for s in both]) / norm[t])
    assert 81 not in c["geowalk"]["dna"] and 81 in c["cmaes_sub"]["dna"]
    assert S["task_diffs"] == pytest.approx(tdm)
    assert S["tasks_cma_better"] == int((np.asarray(tdm) > 0).sum()) and S["n_tasks"] == len(tasks)
    assert S["p_task"] == pytest.approx(stats.wilcoxon_exact_p(np.asarray(tdm)))
    own = [np.mean([v for v, _ in c["geowalk"][t].values()]) / norm[t]
           - np.mean([v for v, _ in c["cmaes_sub"][t].values()]) / norm[t] for t in tasks]
    assert max(abs(a - b) for a, b in zip(own, tdm)) > 1e-6             # the fixture tells the conventions apart


# ---- LassoBench: one objective for the rerun and the published set (review S7) ---------------------------------
def _pub_ctx(tmp_path, monkeypatch, cold=None):
    import v5_fixtures
    from test_lasso_cr import write_original_design
    R = write_original_design(tmp_path / "lassobench_cr")
    R.papenmeier_leukemia(range(43, 82))
    if cold is not None:                                   # the published set evaluates the cold start differently
        monkeypatch.setattr(v5_fixtures, "cold_scores", cold)
    pub = tmp_path / "lassobench_pub"
    write_pub_design(pub)
    p = tmp_path / "v" / "verdicts.json"
    write_verdicts(p, monkeypatch=monkeypatch)
    ctx = _ctx(tmp_path, p)
    ctx.extra.update(lasso_cr_dir=R.root, lasso_pub_dir=str(pub))
    return ctx


def test_objective_check_counts_the_shared_cold_starts(tmp_path, monkeypatch):
    ctx = _pub_ctx(tmp_path, monkeypatch)
    P = stored.lasso_pub_analysis(ctx)
    chk = P["objective_check"]
    # every run of the six cold-start-sharing methods is compared with GeoWalk on its (task, seed); BAxUS and CMA-ES
    # are not; the pilots that timed out have no file
    n = sum(1 for (m, t, s) in P["loaded"]["cold"] if m in lasso_cr.COLD_SHARED)
    assert chk["compared"] == n > 1000 and chk["exact"] == n and chk["beyond_tol"] == 0
    assert "baxus_official" not in lasso_cr.COLD_SHARED and "cmaes_full_published" not in lasso_cr.COLD_SHARED


def test_objective_check_is_fatal_beyond_the_tolerance(tmp_path, monkeypatch):
    import v5_fixtures
    base = v5_fixtures.cold_scores
    ctx = _pub_ctx(tmp_path, monkeypatch, cold=lambda t, s: [v * 1.01 for v in base(t, s)] if t == "dna" else base(t, s))
    with pytest.raises(lasso_cr.LassoLoadError, match=r"\[objective\].* 240 of \d+ runs evaluate the shared"):
        stored.lasso_pub_analysis(ctx)


def test_objective_check_tolerates_the_last_bits(tmp_path, monkeypatch):
    import v5_fixtures
    base = v5_fixtures.cold_scores
    ctx = _pub_ctx(tmp_path, monkeypatch, cold=lambda t, s: [v * (1 + 1e-9) for v in base(t, s)] if t == "dna" else base(t, s))
    chk = stored.lasso_pub_analysis(ctx)["objective_check"]
    assert chk["within_tol"] == 240 and chk["beyond_tol"] == 0 and chk["exact"] == chk["compared"] - 240


# ---- exit code 137 after the limit is a timeout (review M4) -----------------------------------------------------
def test_rc_137_counts_as_timed_out_only_after_the_limit():
    t0 = "2026-10-15T10:00:00+02:00"
    assert lasso_cr.timed_out({"rc": 124})
    assert lasso_cr.timed_out({"rc": 137, "start": t0, "end": "2026-10-15T11:00:10+02:00", "timeout_s": "3600"})
    assert not lasso_cr.timed_out({"rc": 137, "start": t0, "end": "2026-10-15T10:05:00+02:00", "timeout_s": "3600"})
    assert not lasso_cr.timed_out({"rc": 137, "timeout_s": "3600"})          # no times recorded: failed
    assert not lasso_cr.timed_out({"rc": 1}) and not lasso_cr.timed_out({"rc": 0})


def test_rc_137_after_the_limit_is_a_timeout_in_the_coverage(tmp_path):
    from test_lasso_cr import write_original_design
    R = write_original_design(tmp_path / "lassobench_cr")
    R.papenmeier_leukemia(range(43, 81))
    R.job("full_rerun", "papenmeier_leukemia_s81", ["papenmeier_msr"], "leukemia", 81, {"papenmeier_msr": None},
          rc=137, start="2026-10-15T10:00:00+02:00", end="2026-10-15T11:00:10+02:00")
    c = lasso_cr.load_rerun(R.root)["coverage"][("papenmeier_msr", "leukemia")]
    assert c["timeout"] == [81] and c["failed"] == []


# ---- phase 2 of the published methods ran as the pilot (review M12) ---------------------------------------------
def test_published_phase_two_records_must_match_the_pilot(tmp_path):
    pub = tmp_path / "lassobench_pub"
    write_pub_design(pub)
    mp = os.path.join(str(pub), lasso_cr.PUB_SET, "manifest.jsonl")
    recs = [json.loads(x) for x in open(mp)]
    for r in recs:
        r["clone"] = "/data2/prusek/cr_pubprod/clone"
        r["host"] = "kajman" if r.get("seed") != 50 else "other"      # the host is not part of the rule
    with open(mp, "w") as f:
        f.write("".join(json.dumps(r) + "\n" for r in recs))
    lasso_cr.load_rerun(str(pub), design=lasso_cr.PUB)                   # consistent: loads
    for r in recs:
        if r["label"] == "hvarfner_published_leukemia_s60":
            r["clone"] = "/somewhere/else"
    with open(mp, "w") as f:
        f.write("".join(json.dumps(r) + "\n" for r in recs))
    with pytest.raises(lasso_cr.LassoLoadError, match=r"hvarfner_published leukemia: clone differs from the pilot's "
                                                      r"\(seed 42\) at seeds 60"):
        lasso_cr.load_rerun(str(pub), design=lasso_cr.PUB)


# ---- a camera-ready run comes from a clean, committed generator and the pinned input commits (review S4) --------
def test_camera_ready_run_needs_a_clean_tree_and_the_pinned_commits(tmp_path, monkeypatch, capsys):
    import make_tables

    def stop(*a, **k):
        raise make_tables.ConfigError("stopped by the test after the provenance checks")
    monkeypatch.setattr(make_tables, "load_config", stop)
    pins = [f"--expected-commit={k}={v}" for k, v in make_tables.PINNED_COMMITS.items()]
    head = "c" * 40

    def run(git_out, *argv):
        monkeypatch.setattr(make_tables, "git", lambda *a, **k: git_out.get(a[0], ""))
        rc = make_tables.main(["--outdir", str(tmp_path / f"o{len(argv)}"), "--inputs-root", str(tmp_path), *argv])
        return rc, capsys.readouterr().err

    clean = {"rev-parse": head, "status": ""}
    rc, err = run(clean, *pins)
    assert rc == 2 and "stopped by the test" in err                       # accepted: reaches the config loader
    rc, err = run({"rev-parse": head, "status": " M crlib/tables.py"}, *pins)
    assert rc == 2 and "refused" in err and "dirty (1 paths)" in err
    rc, err = run({"rev-parse": "", "status": ""}, *pins)
    assert rc == 2 and "not a git checkout" in err
    rc, err = run(clean, *[p for p in pins if "CRPUB" not in p], "--expected-commit=CRPUB=" + "e" * 40)
    assert rc == 2 and "differs from the commits pinned" in err and "CRPUB" in err
    rc, err = run({"rev-parse": "", "status": " M x"}, "--expected-commit=CR=" + "a" * 40, "--validation-run")
    assert rc == 2 and "stopped by the test" in err                       # smoke runs and tests: both waived
    assert make_tables.PINNED_COMMITS == {"CR": "14481cb55d21a770da11c3d6f866fcae98891c18",
                                          "CRCPU": "14481cb55d21a770da11c3d6f866fcae98891c18",
                                          "CRPUB": "963ef040cae648d4132e1d3330a2a9c06456e97d",
                                          # the GuacaMol runs of 2026-11-02 (root CRRD, commit 1ea2f7d)
                                          "CRRD": "1ea2f7d8fdbec151c59c91e68bc15f45bc79379e",
                                            "CRC4": "2a899ad301d3cb0badd906adfb4d81ad0b50f168"}


# ---- the Hvarfner arm is chosen once, on the default analysis (second set (e), review S6) -----------------------
def test_hvarfner_arm_is_chosen_once_on_the_default_analysis(tmp_path, write_config, monkeypatch):
    from conftest import arm_cfg, ours_record
    from crlib import TASKS_T11
    from crlib.config import load_config
    from crlib.engine import Engine
    from crlib.loader import load_all
    arms = ("hvarfner_published", "hvarfner_published_allobs")
    for arm, lev in zip(arms, (0.40, 0.42)):
        d = tmp_path / arm
        d.mkdir()
        for t in ("adip", "rano"):
            for s in (42, 43):
                rec = ours_record(arm, t, s, scores=[0.1] * 29 + [lev])
                rec["provenance"]["cr_arm"] = arm
                (d / f"ours_{arm}_{TASKS_T11[t]}_s{s}_128d.json").write_text(json.dumps(rec))
    ds = load_all(load_config(write_config({a: arm_cfg(a, cr_arm=a) for a in arms}, tasks=("adip", "rano"),
                                           seeds="42-43")), "abc1234", log=lambda m: None)
    shared = {}

    def ctx(v):
        return Ctx(ds=ds, eng=Engine(ds), reg=Registry("x", "0" * 64, v), var=VARIANTS[v], outdir=str(tmp_path),
                   script_commit="x", mode="cr", extra={"_shared_cache": shared})
    with pytest.raises(published.PublishedInputError, match="chosen on the default analysis"):
        published.hvarfner_arm(ctx("clean_nofallback"))              # never chosen on a variant
    assert published.hvarfner_arm(ctx("default")) == "hvarfner_published_allobs"
    monkeypatch.setattr(published.stats, "level", lambda v, *a, **k: -1.0)   # a recomputation would pick the other
    v = ctx("clean_nofallback")
    assert published.hvarfner_arm(v) == "hvarfner_published_allobs"
    assert v.reg.records["crPubHvarfnerChosen"]["value"] == "hvarfner_published_allobs"
    assert "crPubHvarfnerInboxMean" not in v.reg.records                  # nothing is recomputed in the variant


# ---- tasks won / lost use the 1e-6 tie rule (RUN_RULES 2026-10-12 (e); review M1) -------------------------------
def test_tasks_won_and_lost_use_the_tie_rule():
    cells = [(t, s) for t in ("adip", "dhop", "fexo") for s in (42, 43)]
    a = {c: 0.5 for c in cells}
    b = {("adip", 42): 0.5 - 5e-7, ("adip", 43): 0.5 - 5e-7,          # adip: ahead by 5e-7, a tie
         ("dhop", 42): 0.4, ("dhop", 43): 0.4,                         # dhop: won
         ("fexo", 42): 0.6, ("fexo", 43): 0.6}                         # fexo: lost
    pr = stats.paired(a, b, cells)
    assert (pr.tasks_won, pr.tasks_lost, pr.n_tasks) == (1, 1, 3)


# ---- LassoBench marks, counts and captions (review S1, S2, M3) --------------------------------------------------
def test_lassobench_no_result_cells_are_nc_and_counts_cover_the_printed_methods(tmp_path, monkeypatch):
    from test_lasso_cr import write_original_design
    R = write_original_design(tmp_path / "lassobench_cr")
    R.papenmeier_leukemia(range(43, 82))
    timeouts = [("hvarfner_published", "synt_hard", s) for s in lasso_cr.SEEDS]        # every seed over the limit
    timeouts += [("papenmeier_published", "dna", s) for s in range(42, 52)]            # frozen (not printed)
    timeouts += [("turbo_published", "dna", s) for s in (42, 43, 44)]
    failed = [("turbo_published", "dna", s) for s in (45, 46)]
    pub = tmp_path / "lassobench_pub"
    write_pub_design(pub, timeout=timeouts, failed=failed)
    p = tmp_path / "v" / "verdicts.json"
    write_verdicts(p, monkeypatch=monkeypatch)
    ctx = _ctx(tmp_path, p)
    ctx.extra.update(lasso_cr_dir=R.root, lasso_pub_dir=str(pub))
    t = stored.tab_lassobench(ctx)
    rows = {r[0]: r for r in t.rows if isinstance(r, list)}
    hv = published.name_or_label(ctx, "hvarfner_published", "lasso")
    assert rows[hv][5] == "n.c." and "pend." not in [x for r in rows.values() for x in r]       # S1: not 'pend.'
    tb = published.name_or_label(ctx, "turbo_published", "lasso")
    full = t.record_tables[0].caption        # RUN_RULES 2026-10-29: the coverage sentences are in the full table
    assert f"{hv} $D=1000$ (40 over the time limit)" in full and "over the time limit" not in t.caption
    # M3: the excluded seeds are attributed to the cause their records state; no blanket 'one-hour limit'
    assert f"{tb}, $D=180$: 35/40 (3 over the time limit, 2 failed)." in full
    assert "Seeds over the one-hour limit" not in full
    # S2: the printed counts cover the eight printed methods (the frozen-noise timeouts are not counted)
    rec = ctx.reg.records
    P = stored.lasso_pub_analysis(ctx)
    cov = P["loaded"]["coverage"]
    # 40 + 3 timeouts above and the three pilots that exceeded the hour (two learned Papenmeier variants, Linear BO)
    assert rec["crLassoPubCellsTimeout"]["value"] == 40 + 3 + 3
    assert rec["crLassoPubCellsExpected"]["value"] == sum(c["expected"] for (m, _), c in cov.items() if m in P["printed"])
    assert P["loaded"]["summary"]["timeout"] == 40 + 3 + 3 + 10          # the full design still counts the frozen ones


def test_verification_status_is_the_same_word_in_every_table():
    """Second pass of 2026-10-29 (C2): Tables 13, 14 and 15 say 'yes', 'yes, with allowance' or 'no' from one rule; the
    all-observations Hvarfner run is a variant whose settings were not verified."""
    import types
    ctx = types.SimpleNamespace(mode="cr", cache={"verdicts": {
        "hvarfner": {"verified": True, "entries": [{"pass": True, "pass_strict": False},
                                                   {"pass": False, "pass_strict": False}]},
        "xu": {"verified": True, "entries": [{"pass": True, "pass_strict": True}]},
        "turbo": {"verified": False, "entries": []}}})
    assert published.recipe_status(ctx, "hvarfner") == "yes, with allowance"
    assert published.recipe_status(ctx, "xu") == "yes" and published.recipe_status(ctx, "turbo") == "no"
    assert published.verification_status(ctx, "hvarfner_published") == "yes, with allowance"
    assert published.verification_status(ctx, "hvarfner_published_allobs") == "no"
    assert published.verification_status(ctx, "xu_published") == "yes"


def test_na_reason_is_derived_from_the_result_files():
    cov = {("xu_published", "synt_hard"): {"not_applicable": [42], "errors": {
        "Xu SBO-Matern (RI) is undefined at D=1000: sqrt(D) lies outside the official lengthscale constraint": 40}}}
    # L5 (RUN_RULES 2026-10-22, wording): not run, why, and that the authors report a 1003-D run
    txt = stored.na_reason(cov, "xu_published")
    assert txt.startswith("In the authors' released configuration") and "$D>900$" in txt and "1003" in txt
    cov[("xu_published", "synt_hard")]["errors"] = {"undefined": 40}
    assert stored.na_reason(cov, "xu_published") == ("The result files of the recipe mark it as not applicable at "
                                                     "that dimension.")


# ---- captions name the family BH actually used (review M6) -------------------------------------------------------
def test_bh_note_names_the_family_size_actually_corrected():
    assert stats.bh_note([0.01, 0.2, 0.5]) == ""
    nan = float("nan")
    assert stats.bh_note([0.01, nan, 0.5], "rows") == (" (1 of the 3 rows has no $p$-value: all per-task differences "
                                                        "zero, or too few tasks; the correction runs over the other 2)")
    assert stats.bh_note([nan, nan]).startswith(" (none of the 2 tests has a $p$-value")
    assert stats.bh([0.01, nan, 0.5])[1] != stats.bh([0.01, nan, 0.5])[1]      # NaN passes through BH uncounted


def test_kajman_is_described_by_the_hardware_its_runs_used():
    """Review M11: the rules state kajman's CPU model only, and its runs are CPU only."""
    assert "L40S" not in baselines.HOSTS["kajman"] and "Xeon Gold 6542Y" in baselines.HOSTS["kajman"]


def test_final_run_stops_on_an_objective_mismatch_before_any_table(tmp_path, monkeypatch):
    import make_tables
    import v5_fixtures
    from test_lasso_cr import write_original_design
    R = write_original_design(tmp_path / "lassobench_cr")
    R.papenmeier_leukemia(range(43, 82))
    base_cold = v5_fixtures.cold_scores
    monkeypatch.setattr(v5_fixtures, "cold_scores", lambda t, s: [v + 0.5 for v in base_cold(t, s)])
    pub = tmp_path / "lassobench_pub"
    write_pub_design(pub)
    p = tmp_path / "v" / "verdicts.json"
    write_verdicts(p, monkeypatch=monkeypatch)

    def stop(*a, **k):
        raise make_tables.ConfigError("stopped by the test after the input checks")
    monkeypatch.setattr(make_tables, "load_config", stop)
    rc = make_tables.main(["--expected-commit", "CR=" + "a" * 40, "--inputs-root", str(tmp_path), "--lasso-cr-root",
                           R.root, "--lasso-pub-root", str(pub), "--verdicts", str(p), "--lasso-require-finished",
                           "--validation-run", "--outdir", str(tmp_path / "o")])
    failed = (tmp_path / "o" / "LOADER_FAILED.txt").read_text()
    assert rc == 2 and failed.startswith("LASSOBENCH OBJECTIVE CHECK FAILED") and " of " in failed
