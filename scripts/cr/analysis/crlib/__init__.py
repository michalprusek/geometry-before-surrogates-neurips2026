"""Camera-ready table/figure/number generator library (NeurIPS 2026 submission 24168).

Modules
-------
config      sources configuration (one directory + one anchored filename pattern per arm)
loader      loads every arm, enforces the failure modes (missing arm/cell, duplicate cell,
            provenance mismatch) and counts non-completed and fallback cells
stats       levels (mean of per-task means, flat mean), paired cell-level statistics,
            Benjamini-Hochberg, task-level exact Wilcoxon
engine      cell masks and the statistics of one comparison, with file-set provenance
numbers     registry of every printed number -> numbers.tex and the dossier
latex       booktabs table writer
tables      the table builders (one function per camera-ready table)
stored      NAS-Bench-101 and LassoBench loaders (non-GuacaMol stored sets)
floor       retrieval floor (vendored authors' module, sha256-checked)
figures     matplotlib figures (pdf.fonttype 42)
queries     small query language shared by the legacy checks and the posted/submitted tables
decomp      the pre-registered decomposition (2 x 3 levels, 9 contrasts, by N, per task), the factorial
            by box, the equivalence (TOST) table, distinct proposals, the GP radius dose-response
geometry    shell geometry of the encoder means (tab_geometry, fig_shells, the inline geometry numbers)
mc          Monte Carlo norms of every candidate support (2,000,000 draws, fixed seeds, SE)
supports    tab_support_geometry and the per-arm reporting card tab_effective_support
baselines   Appendix H baseline cards (ten fields per method) rendered from appendixH_facts.csv, with marks on
            every cell that needs author proofreading
controls    the control block (C1 input transform, C2 anisotropy, C3 replication on seeds 82-121) and its
            pre-committed readings, computed mechanically from the rule text
extras      seed-block permutation / shared-cold-start correlation of the decomposition contrasts, anytime
            metrics, Linear BO disclosure rows
lasso_cr    LassoBench camera-ready rerun: provenance join to manifest.jsonl, coverage (loud), expected design
september   stored September runs: E-1 on its eight extra eligible tasks, selection-rule robustness (p1 blocks)
scope       GEN v5: what a camera-ready run prints (RUN_RULES 2026-10-14): unprinted arms are not loaded,
            unprinted tables not built, variants of unprinted analyses not run
labels      GEN v5: descriptive labels of every baseline arm (print rule of 2026-10-12)
published   GEN v5: published-recipe block: verdicts (pinned verdicts.json), names, the baseline family
            (tab_published) and the verification table (tab_verification)
"""

TASKS_T11 = {
    "adip": "amlodipine_mpo",
    "dhop": "deco_hop",
    "fexo": "fexofenadine_mpo",
    "med1": "median_1",
    "med2": "median_2",
    "osmb": "osimetrinib_mpo",
    "pdop": "perindopril_mpo",
    "rano": "ranolazine_mpo",
    "shop": "scaffold_hop",
    "siga": "sitagliptin_mpo",
    "zale": "zaleplon_mpo",
}
POLI_TO_SHORT = {v: k for k, v in TASKS_T11.items()}
SEEDS_40 = list(range(42, 82))
