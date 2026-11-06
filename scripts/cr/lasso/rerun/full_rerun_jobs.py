#!/usr/bin/env python3
"""Job list for the full LassoBench rerun (RUN_RULES amendment 2026-10-12). One job per
(method-group, task, seed); ordered by seed so every task progresses together."""
tasks = ["synt_simple", "synt_medium", "dna", "synt_high", "synt_hard"]
groups = [("cheap", "random,cmaes_sub,geowalk,cmaes_full"), ("turbo", "turbo"), ("baxus", "baxus"),
          ("hvarfner", "hvarfner_gp"), ("papenmeier", "papenmeier_msr"), ("xu", "xu_standard_gp"),
          ("linbo", "linear_bo")]
R = "/path/to/NeurIPS-cr-ana/scripts/run_lassobench.py"
C = "/path/to/cr_analysis/code/run_lasso.sh"
# leukemia pilots first (they decide what is reported at D = 7129)
for name, m in (("papenmeier", "papenmeier_msr"), ("xu", "xu_standard_gp")):
    print(f"bash {C} full_rerun pilot_{name}_leukemia_s42 results {R} {m} leukemia 42 3600")
for seed in range(42, 82):
    for task in tasks:
        for name, m in groups:
            print(f"bash {C} full_rerun {name}_{task}_s{seed} results {R} {m} {task} {seed} 3600")
    print(f"bash {C} full_rerun hvarfner_leukemia_s{seed} results {R} hvarfner_gp leukemia {seed} 3600")
