"""Write jobs_primary_r1.txt: one line per verification run, queue order = pre-registered priority
(RUN_RULES revision 1): expected-longest first, frozen-noise before learned-noise, Hvarfner 30 before 41.
Fields: recipe benchmark seed budget batch n_init m_d   ('-' = harness default / not passed)."""
import sys
L = []
def add(recipe, bench, seeds, budget, batch, n_init, m_d="-"):
    for s in range(seeds):
        L.append(f"{recipe} {bench} {s} {budget} {batch} {n_init} {m_d}")
# block 1: Papenmeier, frozen noise (reproduce_fig8.sh default), then learned noise; interleaved by seed is not
# needed: all 40 are expected-longest (8.5-75 h per run)
add("msr", "lasso_dna", 10, 1000, 1, 40)
add("mle_scaled", "lasso_dna", 10, 1000, 1, 40)
add("msr_noise_learned", "lasso_dna", 10, 1000, 1, 40)
add("mle_scaled_noise_learned", "lasso_dna", 10, 1000, 1, 40)
# block 2: Hvarfner Fig. 7, 30-point design (paper text, figure) then 41 (released code default)
add("hvarfner", "lasso_dna", 20, 1000, 1, 30)
add("hvarfner", "lasso_dna", 20, 1000, 1, 41)
# block 3: Doumont Fig. 2 (n_reps '>=10' -> 10 seeds, summarize --n-reps doumont_linear__lasso_dna__fig2=10)
add("doumont", "lasso_dna", 10, 1000, 1, 30)
# block 4: Xu Fig. 6 (offset 1 -> needs 301; the release runs 320)
add("xu", "lasso_dna", 10, 320, 1, 20)
# block 5: BAxUS Fig. 3, both panels gate the label; m_D 1000; last record n = 1000 at offset 1 -> 1001
add("baxus", "lasso_synt_hard", 20, 1001, 1, 10, 1000)
add("baxus", "lasso_synt_high", 20, 1001, 1, 10, 1000)
# block 6: TuRBO (LassoBench files, q = 10), n_init 0.1 d LHS
add("turbo_published", "lasso_synt_simple", 30, 500, 10, 6)
add("turbo_published", "lasso_synt_medium", 30, 500, 10, 10)
# block 7: CMA-ES (LassoBench files), x0 = 0, f(x0) outside the budget; DNA informative
for b in ("lasso_synt_simple", "lasso_synt_medium", "lasso_synt_high", "lasso_synt_hard", "lasso_dna"):
    add("cmaes", b, 30, 500, 1, "-")
out = sys.argv[1]
with open(out, "w") as f:
    f.write("# recipe benchmark seed budget batch n_init m_d -- RUN_RULES 2026-10-13 revision 1; queue order = priority\n")
    f.write("\n".join(L) + "\n")
print(len(L), "jobs ->", out)
