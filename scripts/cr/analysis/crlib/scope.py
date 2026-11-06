"""What a camera-ready run prints, and what it therefore never reads (RUN_RULES amendment 2026-10-14, "Scope of the
camera-ready reduced to the promised content", with its correction of 11:25 and the rows of 2026-10-19).

The camera-ready prints what the authors promised, the corrections without which a printed sentence would be false,
and the published-recipe block. Everything else is neither built nor loaded in camera-ready mode, so no result of an
unprinted arm is ever read: its arms are removed from the sources configuration before the loader opens a file, its
table builders are not called, and the analysis variants that exist only for it are not run. Legacy mode
(``--legacy``) keeps every builder, because it validates the code paths against numbers verified earlier and reads
only stored files.

Every name below is checked against the configuration and the builder lists by ``check_scope`` (a typo would
otherwise silently keep an arm in the run).
"""
from __future__ import annotations

RULE = "RUN_RULES amendment 2026-10-14 (scope), correction of 11:25, rows of 2026-10-19"

#: Table builders (function names) whose tables are not printed (scope items 1-10).
NOT_PRINTED_TABLES = {
    # (1) control block C1-C3 and C4
    "tab_controls": "control block C1-C2 (scope item 1)",
    "tab_replication": "control block C3, seeds 82-121 (scope item 1)",
    # (2) the 2x3 decomposition; the paper prints the 2x2 factorial of Table 2 instead
    "tab_decomposition": "2x3 decomposition (scope item 2)",
    "tab_decomposition_contrasts": "2x3 decomposition contrasts (scope item 2)",
    "tab_decomposition_contrasts_byN": "2x3 decomposition contrasts at N = 10, 20 (scope item 2)",
    "tab_decomposition_pertask": "2x3 decomposition per task (scope item 2)",
    "tab_decomposition_robustness": "robustness of the 2x3 contrasts (scope item 2)",
    "tab_factorial_details": "factorial in every generation (scope item 2: Table 2 prints the five contrasts)",
    "tab_distinct": "distinct proposals (scope item 2, analysis plan 2.5)",
    # (3) equivalence (TOST)
    "tab_equivalence": "TOST (scope item 3)",
    # (4) training-corpus geometry: tab_geometry and fig_shells keep ZINC-250k only (see GEOMETRY_*)
    # (5) E-1 extra tasks
    "tab_e1_extra_tasks": "E-1 extra-task block (scope item 5)",
    "tab_selection_rule": "September p1 selection-rule blocks (scope: not promised)",
    # (6) retrieval floor
    "tab_floor": "retrieval floor (scope item 6)",
    # (7) GP radius arms
    "tab_gp_radius": "GP radius dose-response (scope item 7)",
    "tab_shell_prior": "N(0, I) and uniform-box candidate controls (scope: not promised)",
    # (8) anytime and regret
    "tab_anytime": "anytime and cumulative regret (scope item 8)",
    # (9) box sensitivity
    "tab_box_sensitivity": "box sensitivity incl. wave 3b (scope item 9)",
    "tab_factorial_by_box": "factorial with two boxes (scope item 9)",
    # (10) posted against printed
    "tab_posted_vs_printed": "full posted-vs-printed table (scope item 10: a short changes paragraph instead)",
    # backlog scope note of 2026-10-14 11:30
    "tab_budget_fidelity": "budget fidelity (scope note 11:30)",
    "tab_effective_support": "effective support (scope note 11:30)",
    "tab_support_geometry": "support geometry (scope note 11:30)",
    "tab_linear_bo": "Linear BO task-level table (scope note 11:30)",
}

#: Tables built only as a repository record (written to records/, never next to the printed tables, never in the
#: LaTeX check): RUN_RULES 2026-10-19 (e). Building it also attaches the submitted values to the dossier.
RECORD_ONLY_TABLES = {
    "tab_changes_vs_submitted": "submitted against printed values (RUN_RULES 2026-10-19 (e); scope item 10): a record",
    # RUN_RULES 2026-10-24: Appendix H prints the overview (tab_baselines_overview); the cards and the run status
    # are repository records (records/ of the output folder, committed as docs/baselines/)
    "tab_baselines": "the Appendix H cards (RUN_RULES 2026-10-24): a repository record, docs/baselines/BASELINE_CARDS.md",
    "tab_arm_status": "the run status of every method (RUN_RULES 2026-10-24): a repository record, "
                      "docs/baselines/RUN_STATUS.md",
    # RUN_RULES 2026-10-29 (2): tables that the text does not need are repository records
    "tab_compute": "the compute table of Appendix K (RUN_RULES 2026-10-29 (2)): a record; its totals stay in the text",
    "tab_radius_pertask": "the per-task radius table (RUN_RULES 2026-10-29 (2)): a record",
}

#: RUN_RULES 2026-10-29 (2): tables of Appendix I with the same unit are printed as blocks of one table, each block with
#: its own family. The builder of a merged table still runs and registers every number under its own table name
#: (numbers.tex keeps its order and bytes); its blocks are appended to the host table, which keeps its own label.
MERGED_TABLES = {
    "tab_estimators": ("tab_radius", "radius estimator block of the radius table (RUN_RULES 2026-10-29 (2))"),
    "tab_shells": ("tab_radius", "thick-shell block of the radius table (RUN_RULES 2026-10-29 (2))"),
    "tab_pca_directions": ("tab_convex", "last block of the convex-combination table (RUN_RULES 2026-10-29 (2))"),
}

#: RUN_RULES 2026-10-29 (2): figures that the text does not need, written to records/ instead of next to the tables
RECORD_ONLY_FIGURES = {
    "fig_lassobench": "the LassoBench figure (RUN_RULES 2026-10-29 (2)): a record",
    "fig_trajectories": "the trajectory figure (RUN_RULES 2026-10-29 (2)): a record; its values are in the trajectory "
                        "table and the convergence figure",
}

#: Number functions of make_tables.run_variant that only feed unprinted tables.
NOT_PRINTED_NUMBERS = {
    "anytime_legacy_numbers": "anytime metrics (scope item 8)",
    "extra_numbers": "budget-fidelity numbers (scope note 11:30)",
}

#: Arms that are not loaded in camera-ready mode: every reader of these arms is an unprinted table (measured with
#: an instrumented smoke run of c4f68f0: tab_arm_status and tab_baselines list every loaded arm and are not
#: readers in this sense).
NOT_LOADED_ARMS = {
    # (1) control block (tag cr-controls-2026-10-12)
    "hvarfner_scaled_gp_box_argmax_raw": "control C1", "hvarfner_scaled_gp_boxdir_argmax": "control C2a",
    "random_boxdir": "control C2b", "random_search__s82": "control C3", "random_box_faithful__s82": "control C3",
    "hvarfner_scaled_gp__s82": "control C3", "hvarfner_scaled_gp_box_argmax__s82": "control C3",
    "hvarfner_scaled_gp_sphere_continuous__s82": "control C3", "hvarfner_scaled_gp_box_faithful__s82": "control C3",
    # (7) GP radius arms
    "hvarfner_scaled_gp_sphere05": "GP radius arm (scope item 7)",
    "hvarfner_scaled_gp_sphere15": "GP radius arm (scope item 7)",
    "hvarfner_scaled_gp_sphere20": "GP radius arm (scope item 7)",
    # (9) wave 3b (stopped 2026-10-14 11:10, incomplete by design) and wave 4 (not run)
    "hvarfner_scaled_gp_box": "wave 3b, +-3 sigma box (scope item 9; stopped)",
    "papenmeier_msr_box": "wave 3b, +-3 sigma box (scope item 9; stopped)",
    "xu_standard_gp_box": "wave 3b, +-3 sigma box (scope item 9; stopped)",
    "hvarfner_scaled_gp_norm_box_argmax": "wave 3b, [min,max] arg-max twin (scope item 9; stopped)",
    "papenmeier_msr_published_norm_box": "wave 4 (not run; scope item 9)",
    # (9) the [min,max] box variant of the GP recipes (box sensitivity, variant R2)
    "hvarfner_scaled_gp_norm_box": "[min,max] box (scope item 9)",
    "papenmeier_msr_norm_box": "[min,max] box (scope item 9)",
    "xu_standard_gp_norm_box": "[min,max] box (scope item 9)",
    # (2) arms of the 2x3 decomposition and of the shell-prior controls only
    "random_box_faithful": "no-surrogate box cell of the 2x3 decomposition (scope item 2)",
    "hvarfner_scaled_gp_prior": "N(0, I) candidate control (tab_shell_prior)",
    "random_search_prior": "N(0, I) candidate control (tab_shell_prior)",
    # legacy stand-in for the published MSR maximiser; no printed table reads it
    "papenmeier_msr_raasp": "July RAASP arm (legacy substitute only)",
    # stored arms read only by unprinted tables
    "e1_hvarfner_scaled_gp": "E-1 generation (tab_factorial_details, tab_factorial_by_box)",
    "e1_hvarfner_scaled_gp_sphere_continuous": "E-1 generation (tab_factorial_details, tab_factorial_by_box)",
    "e1_hvarfner_scaled_gp_box_argmax": "E-1 generation (tab_factorial_details, tab_factorial_by_box)",
    "e1_hvarfner_scaled_gp_box_faithful": "E-1 generation (tab_factorial_details, tab_factorial_by_box)",
    "linear_bo_sphere": "stored Linear BO on the sphere (tab_linear_bo)",
}

#: Analysis variants that exist only for an unprinted analysis (they would read excluded arms).
NOT_RUN_VARIANTS = {
    "R2_minmax": "the [min,max] box (box sensitivity, scope item 9)",
    "R4_E1": "factorial from the E-1 rerun (tab_factorial_details is not printed; Table 2 prints the camera-ready runs)",
}

#: The posted generation (namespace 'posted', sources_legacy.json) is read only by unprinted tables
#: (tab_posted_vs_printed, tab_factorial_by_box, tab_factorial_details): its include is not loaded.
SKIP_INCLUDES = {"posted": "posted generation (read only by unprinted correspondence and factorial tables)"}

#: tab_geometry and fig_shells: ZINC-250k only (scope item 4: no training-corpus geometry, no posterior-sample shell).
GEOMETRY_COLUMNS = ("default", "strip")
GEOMETRY_ROWS_DROPPED = ("SumSigma", "ZNorm")        # posterior variance and posterior-sample norm rows


def check_scope(arm_names, builder_names, variant_names, number_names, figure_names=None) -> list[str]:
    """Names in the scope lists that do not exist (an empty list = the lists are consistent)."""
    bad = [f"arm {a}" for a in NOT_LOADED_ARMS if a not in arm_names]
    bad += [f"table builder {b}" for b in list(NOT_PRINTED_TABLES) + list(RECORD_ONLY_TABLES) if b not in builder_names]
    bad += [f"merged table {b} -> {h}" for b, (h, _) in MERGED_TABLES.items()
            if b not in builder_names or h not in builder_names or h in MERGED_TABLES]
    if figure_names is not None:
        bad += [f"figure builder {f}" for f in RECORD_ONLY_FIGURES if f not in figure_names]
    bad += [f"variant {v}" for v in NOT_RUN_VARIANTS if v not in variant_names]
    bad += [f"number function {f}" for f in NOT_PRINTED_NUMBERS if f not in number_names]
    return bad
