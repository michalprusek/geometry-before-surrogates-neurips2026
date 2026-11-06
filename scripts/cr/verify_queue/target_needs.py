import sys, json
sys.path.insert(0, sys.argv[1])
from scripts.cr import verify_baselines as vb
T = vb.load_targets(vb.TARGETS_FILE)
pairs = [("hvarfner","lasso_dna"),("msr","lasso_dna"),("mle_scaled","lasso_dna"),("msr_noise_learned","lasso_dna"),
 ("mle_scaled_noise_learned","lasso_dna"),("xu","lasso_dna"),("doumont","lasso_dna"),("baxus","lasso_synt_hard"),
 ("baxus","lasso_synt_high"),("turbo_published","lasso_synt_simple"),("turbo_published","lasso_synt_medium"),
 ("turbo_published","lasso_synt_high"),("turbo_published","lasso_synt_hard"),("turbo_published","lasso_dna"),
 ("cmaes","lasso_synt_simple"),("cmaes","lasso_synt_medium"),("cmaes","lasso_synt_high"),("cmaes","lasso_synt_hard"),("cmaes","lasso_dna")]
for rc, b in pairs:
    tid = vb.default_target_id(rc, b); t = T.get(tid)
    if t is None: print(rc, b, tid, "NO TARGET"); continue
    off = vb._num(t.get("x_eval_offset")) or 0.0
    recs = [r for recs in t["_norm"].values() for r in recs]
    need = max(((r["x_logged"] if r["x_logged"] is not None else r["n"]) + off) for r in recs)
    gates = [ (r["n"], r["gate"]) for r in recs]
    print(f"{rc:26s} {b:18s} {tid:48s} n_reps={t.get('n_reps')} n_init={t.get('n_init')} q={t.get('batch_size')} off={off} need={need} gate={t.get('gate')} doe<{t.get('doe_only_below')} ns={[g[0] for g in gates]} gating={[g[0] for g in gates if g[1]]}")
