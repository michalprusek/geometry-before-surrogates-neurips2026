# Control C4: L-BFGS-B on the shell, read as the run rules of 2026-10-14 prescribe

> Note of 2026-11-06 (RUN_RULES 2026-11-06 16:30): on the authors' decision the L-BFGS-B arm of this control is the shell x continuous cell of the printed factorial (Table 3 and the contrasts of Appendix J) in place of projected Adam; the sentence below that the printed factorial is unchanged describes the state of 2026-11-04. The values printed since 2026-11-06 are those of the section "The factorial with L-BFGS-B as the continuous maximiser on both supports".


Read on 2026-11-04 at the authors' request, after every final run had been read (RUN_RULES row 2026-11-04). The reading, its contrasts, test and thresholds are those of RUN_RULES row 2026-10-14, fixed before the control ran. The printed factorial (Table 2) is unchanged; this record is the only place of the result.

## Cells

- shell, candidate-set argmax: 440 completed cells, 1 fallback step(s)
- shell, projected Adam: 440 completed cells, 1 fallback step(s)
- shell, L-BFGS-B (C4): 440 completed cells, 1 fallback step(s)
- box, candidate-set argmax: 440 completed cells, 1 fallback step(s)
- box, L-BFGS-B: 440 completed cells, 1 fallback step(s)
- matched set of the printed factorial: 440 cells; with the control: 440 cells

## Check: the printed factorial, as Table 2 prints it

| Cell | Level | Printed |
|---|---|---|
| shell, candidate-set argmax | 0.4228 | 0.4228  |
| shell, projected Adam | 0.3883 | 0.3883  |
| box, candidate-set argmax | 0.3925 | 0.3925  |
| box, L-BFGS-B | 0.3989 | 0.3989  |

| Contrast | n | Delta | Tasks | p_task | d | p_cell | W/L | Printed (Delta, tasks, p_task) |
|---|---|---|---|---|---|---|---|---|
| shell minus box, candidate-set argmax | 440 | +0.0303 | 9/11 | 0.01367 | +0.41 | 8.9e-21 | 275/104 | +0.0303, 9/11, 0.014  |
| shell (Adam) minus box (L-BFGS-B) | 440 | -0.0106 | 5/11 | 0.3203 | -0.12 | 0.092 | 185/204 | -0.0106, 5/11, 0.32  |
| candidate-set argmax minus projected Adam, shell | 440 | +0.0345 | 9/11 | 0.01367 | +0.40 | 1.7e-15 | 267/128 | +0.0345, 9/11, 0.014  |
| candidate-set argmax minus L-BFGS-B, box | 440 | -0.0064 | 1/11 | 0.004883 | -0.08 | 0.0033 | 150/223 | -0.0064, 1/11, 0.0049  |
| interaction I_A (Adam on the shell) | 440 | +0.0409 | 10/11 | 0.00293 | +0.36 | 2.5e-14 | 270/139 | +0.0409, 10/11, 0.0029  |

Share of distinct scores, printed factorial: shell Adam 70.4 % (printed 70.4), other cells 92.9 to 99.3 % (printed 92.9 to 99.3).
Convention check: passed.

## The control on the matched set of all five cells

| Cell | Level | Distinct scores |
|---|---|---|
| shell, candidate-set argmax | 0.4228 | 99.3 % |
| shell, L-BFGS-B (C4) | 0.3999 | 82.2 % |
| box, L-BFGS-B | 0.3989 | 92.9 % |
| box, candidate-set argmax | 0.3925 | 97.0 % |
| shell, projected Adam | 0.3883 | 70.4 % |

### Pre-committed reading (RUN_RULES 2026-10-14): the interaction with L-BFGS-B in place of Adam

| Interaction | n | Delta | Tasks > 0 | p_task | d | p_cell | W/L |
|---|---|---|---|---|---|---|---|
| I_A: (shell argmax - box argmax) - (shell Adam - box L-BFGS-B) | 440 | +0.0409 | 10/11 | 0.00293 | +0.36 | 2.5e-14 | 270/139 |
| I_L: (shell argmax - box argmax) - (shell L-BFGS-B - box L-BFGS-B) | 440 | +0.0294 | 9/11 | 0.004883 | +0.26 | 4.4e-08 | 248/157 |

I_L has the sign of I_A on 10 of the 11 tasks, and its task-level p is 0.004883. The rule asks for at least 8 of 11 and p < 0.05: the control passes, so the interaction does not depend on the optimiser of the shell continuous cell.

| Task | I_A | I_L |
|---|---|---|
| amlodipine_mpo | +0.0402 | +0.0504 |
| deco_hop | +0.0101 | +0.0083 |
| fexofenadine_mpo | +0.0350 | +0.0422 |
| median_1 | -0.0038 | -0.0035 |
| median_2 | +0.0004 | -0.0036 |
| osimetrinib_mpo | +0.0472 | +0.0458 |
| perindopril_mpo | +0.0932 | +0.0468 |
| ranolazine_mpo | +0.1466 | +0.0637 |
| scaffold_hop | +0.0188 | +0.0126 |
| sitagliptin_mpo | +0.0218 | +0.0170 |
| zaleplon_mpo | +0.0409 | +0.0432 |

### Descriptive contrasts of the rule (outside the Benjamini-Hochberg family)

| Contrast | n | Delta | Tasks | p_task | d | p_cell | W/L |
|---|---|---|---|---|---|---|---|
| shell L-BFGS-B minus shell Adam (optimiser at fixed support) | 440 | +0.0116 | 7/11 | 0.3652 | +0.14 | 0.0017 | 199/154 |
| shell L-BFGS-B minus box L-BFGS-B (support under the same maximiser) | 440 | +0.0010 | 6/11 | 0.8984 | +0.01 | 0.16 | 212/174 |
| shell argmax minus shell L-BFGS-B | 440 | +0.0229 | 9/11 | 0.01367 | +0.27 | 4.6e-07 | 239/153 |

### The factorial with L-BFGS-B as the continuous maximiser on both supports (not printed)

| Contrast | n | Delta | Tasks | p_task | p_task_BH | d | p_cell | W/L |
|---|---|---|---|---|---|---|---|---|
| support under candidate-set argmax (shell - box) | 440 | +0.0303 | 9/11 | 0.01367 | 0.0171 | +0.41 | 8.9e-21 | 275/104 |
| support under continuous maximisation (shell - box, L-BFGS-B) | 440 | +0.0010 | 6/11 | 0.8984 | 0.898 | +0.01 | 0.16 | 212/174 |
| maximiser on the shell (argmax - L-BFGS-B) | 440 | +0.0229 | 9/11 | 0.01367 | 0.0171 | +0.27 | 4.6e-07 | 239/153 |
| maximiser in the box (argmax - L-BFGS-B) | 440 | -0.0064 | 1/11 | 0.004883 | 0.0122 | -0.08 | 0.0033 | 150/223 |
| I_L: (shell argmax - box argmax) - (shell L-BFGS-B - box L-BFGS-B) | 440 | +0.0294 | 9/11 | 0.004883 | 0.0122 | +0.26 | 4.4e-08 | 248/157 |
