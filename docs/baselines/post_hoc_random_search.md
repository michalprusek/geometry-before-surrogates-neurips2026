# Random search on the shell against the rows of Table 1 that propose off the shell (post-hoc)

Added on 2026-11-03 at the authors' request, after every final run had been read (RUN_RULES row 2026-11-03); not a pre-specified family. Exact two-sided Wilcoxon signed-rank test on the 11 per-task mean differences of best@30 on the matched set of Table 1; Delta = mean of the per-task mean differences; Tasks = tasks on which the first method has the higher mean; p_BH = Benjamini-Hochberg over the seven rows of the second table.

## Check: GeoWalk minus each row, as Table 1 prints it

| Row | n | Delta | Tasks | p_task |
|---|---|---|---|---|
| CMA-ES | 440 | +0.0200 | 8/11 | 0.123 |
| TuRBO | 440 | +0.0437 | 9/11 | 0.03223 |
| BAxUS | 439 | +0.0320 | 9/11 | 0.02441 |
| Vanilla BO | 440 | +0.0218 | 9/11 | 0.08301 |
| Papenmeier MSR | 440 | +0.0327 | 9/11 | 0.01855 |
| Xu Standard GP | 440 | +0.0275 | 9/11 | 0.04199 |
| Linear BO | 440 | +0.0125 | 10/11 | 0.03223 |
| Random Search (shell) | 440 | +0.0108 | 11/11 | 0.0009766 |

## Random search on the shell minus each row that proposes off the shell

| Row | n | Delta | Tasks | p_task | p_BH |
|---|---|---|---|---|---|
| CMA-ES | 440 | +0.0092 | 8/11 | 0.2402 | 0.3247 |
| TuRBO | 440 | +0.0328 | 8/11 | 0.06738 | 0.2358 |
| BAxUS | 439 | +0.0214 | 8/11 | 0.1748 | 0.3247 |
| Vanilla BO | 440 | +0.0110 | 7/11 | 0.3652 | 0.3652 |
| Papenmeier MSR | 440 | +0.0219 | 9/11 | 0.06738 | 0.2358 |
| Xu Standard GP | 440 | +0.0167 | 8/11 | 0.2402 | 0.3247 |
| Linear BO | 440 | +0.0017 | 8/11 | 0.2783 | 0.3247 |
