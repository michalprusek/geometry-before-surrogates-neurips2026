#!/usr/bin/env python3
"""Job lists for the LassoBench published-method production (RUN_RULES 2026-10-12 (3), revision 1 (f)).
One process per (method, task, seed), one hour each. Phase 1: the ten leukemia pilots (seed 42) first,
then the five tasks ordered by seed so every task progresses together. Phase 2 (pub_phase2.sh): leukemia
seeds 43-81 for each method whose pilot exited 0 inside the hour; decided from exit codes only.
usage: pub_jobs.py phase1 | phase2 <method> [<method> ...] | methods"""
import sys
METHODS = ["hvarfner_published", "papenmeier_published", "papenmeier_mle_scaled_published",
           "papenmeier_published_learned", "papenmeier_mle_scaled_published_learned",
           "xu_published", "linear_bo_published", "turbo_published", "baxus_official",
           "cmaes_full_published"]
TASKS = ["synt_simple", "synt_medium", "dna", "synt_high", "synt_hard"]
R = "/path/to/NeurIPS-cr-pubprod/scripts/run_lassobench.py"
C = "/path/to/cr_pubprod/code/run_lasso.sh"
G = "pub_methods"
if sys.argv[1] == "phase1":
    for m in METHODS:
        print(f"bash {C} {G} pilot_{m}_leukemia_s42 results {R} {m} leukemia 42 3600")
    for seed in range(42, 82):
        for task in TASKS:
            for m in METHODS:
                print(f"bash {C} {G} {m}_{task}_s{seed} results {R} {m} {task} {seed} 3600")
elif sys.argv[1] == "phase2":
    for seed in range(43, 82):
        for m in sys.argv[2:]:
            assert m in METHODS, m
            print(f"bash {C} {G} {m}_leukemia_s{seed} results {R} {m} leukemia {seed} 3600")
elif sys.argv[1] == "methods":
    print(" ".join(METHODS))
else:
    raise SystemExit(__doc__)
