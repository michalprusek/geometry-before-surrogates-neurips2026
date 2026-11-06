# Third-party notices

The authors' own code is covered by `LICENSE`. This file lists every third-party component that this
repository contains, downloads or depends on, with its version, licence and source. Licence texts are in
`LICENSES/`; the vendored directories also carry their upstream `LICENSE` file.

## 1. Third-party code contained in this repository

| component | where | version compared | licence | source | changes |
|---|---|---|---|---|---|
| GaBOtorch (N. Jaquier, L. Rozo) | `src/external/gabotorch_bm/` | commit `884f64a9` (the commit that was vendored was not recorded; the comparison is against this one) | MIT, Copyright (c) 2020 Noémie Jaquier: `src/external/gabotorch_bm/LICENSE` | https://github.com/NoemieJaquier/GaBOtorch | 27 of 52 files identical; 25 modified: package imports rewritten from `BoManifolds.*` to `src.external.gabotorch_bm.*`, and further edits in `manifold_optimization/manifold_optimize.py`, `manifold_gp_fit.py`, `numpy_list_converter.py` and `pymanopt_addons/tools/autodiff/_tensorflow.py` |
| MaternGaBO (N. Jaquier, V. Borovitskiy, A. Smolensky, A. Terenin, T. Asfour, L. Rozo) | `src/external/matern_gabo_bm/` | commit `84bffda3` (as above) | MIT, Copyright (c) 2021 Noémie Jaquier: `src/external/matern_gabo_bm/LICENSE` | https://github.com/NoemieJaquier/MaternGaBO | 37 of 50 files identical; 13 modified: package imports rewritten, and further edits in `kernel_utils/kernels_sphere.py`, `kernels_spd.py`, `kernels_hyperbolic.py`, `manifold_optimization/manifold_optimize.py`, `Riemannian_utils/spd_utils_torch.py`, `spd_constraints_utils_torch.py`, `euclidean_optimization/euclidean_constrained_optimize.py` and `pymanopt_addons/tools/autodiff/_tensorflow.py` |
| pymanopt add-ons | `src/external/gabotorch_bm/pymanopt_addons/`, `src/external/matern_gabo_bm/pymanopt_addons/` | as distributed inside the two packages above; their `README.txt` names the fork https://github.com/leonbottou/pymanopt | BSD-3-Clause, Copyright (c) 2015-2016 Pymanopt Developers: `pymanopt_addons/LICENSE` | https://github.com/pymanopt/pymanopt | see the two rows above |
| linear-bo, `SphericalLinearKernel` (C. Doumont et al.) | `src/spherevo/linear_bo_kernel.py` | commit `6ee3bb68` | MIT, Copyright (c) 2025 Colin Doumont: `LICENSES/linear-bo-MIT.txt` | https://github.com/colmont/linear-bo | `jaxtyping` annotations removed, `match` rewritten as `if`/`elif`, docstrings reflowed; no kernel mathematics changed |
| LOL-BO, SELFIES Transformer VAE (N. Maus et al.) | `src/codec/vae_model.py`; the vocabulary and tokenizer in `src/codec/selfies_vae.py`; the VAE classes derived from it in `src/codec/spcauchy_vae.py`, `src/codec/vmf_vae.py`, `src/codec/power_spherical_vae.py` | licence file at commit `08761478` | MIT, Copyright (c) 2022 Natalie Maus: `LICENSES/lolbo-MIT.txt` | https://github.com/nataliemaus/lolbo | re-arranged into this package; the other three files hold VAE classes for other latent distributions that reuse parts of it |
| arch2vec (S. Yan et al.) | model components inlined in `scripts/train_arch2vec.py` | licence file at commit `ea01b0cf` (the commit the code was taken from was not recorded) | Apache-2.0: `LICENSES/arch2vec-Apache-2.0.txt`. Upstream has no NOTICE file. | https://github.com/MSU-MLSys-Lab/arch2vec | inlined into one training script and modified |
| BoTorch tutorials (TuRBO-1, BAxUS) | `src/spherevo/turbo_baseline.py`, `scripts/run_turbo_budget.py`, `scripts/run_baxus_budget.py` follow the tutorials; `BaxusPublished` in `src/spherevo/published_external.py` is a port of the BAxUS tutorial (`tutorials/baxus`) | BoTorch 0.17.2 is the version of the run environment | MIT, Copyright (c) Meta Platforms, Inc. and affiliates: `LICENSES/botorch-MIT.txt` | https://github.com/meta-pytorch/botorch | own implementations that follow the tutorials (see the module docstrings); the BAxUS port runs as an ask/tell optimiser on our problems |
| TuRBO (D. Eriksson, M. Pearce, J. Gardner, R. Turner, M. Poloczek) | `Turbo1Published` and its helpers in `src/spherevo/published_external.py`, used by the published-recipe TuRBO row of the paper | commit `de0db39` | **Uber Non-Commercial License**, Copyright (c) 2019 Uber Technologies, Inc.: `LICENSES/turbo-LICENSE.txt`. The whole file `src/spherevo/published_external.py` is distributed under this licence, for non-commercial purposes only; the MIT licence of this repository does not apply to it. | https://github.com/uber-research/TuRBO | port of `turbo/turbo_1.py` and `turbo/gp.py`, changed on 2026-10-13: rewritten as an ask/tell optimiser; an initial design evaluated outside the optimiser can replace the Latin hypercube design; the search runs on the unit cube of a fixed problem box; a failed step proposes a uniform point instead of ending the run (the file header says the same) |

Under the TuRBO licence, the results of the published-recipe TuRBO runs (the `turbo_published` result files of the
results bundle) are outputs of the licensed work and may not be used for commercial purposes.

`scripts/cr/analysis/crlib/_vendor/zinc_retrieval_floor.py` is the authors' own module, copied from another
branch of their repository and pinned by its sha256; it is not third-party code.

## 2. Methods that are cited but whose released code is NOT in this repository

| method | released code | licence of that code | what this repository contains |
|---|---|---|---|
| TuRBO (Eriksson et al., 2019), apart from the port in section 1 | https://github.com/uber-research/TuRBO | Uber non-commercial licence | BoTorch-based re-implementations (`src/spherevo/turbo_baseline.py`, `src/spherevo/turbo_box_faithful.py`) |
| BAxUS (Papenmeier et al., 2022) | https://github.com/lpapenme/BAxUS | Uber non-commercial terms plus a BSD section | a script that follows the BoTorch tutorial (`scripts/run_baxus_budget.py`) and the port of that tutorial in section 1; no code of the `BAxUS` package |
| Vanilla BO in high dimensions (Hvarfner et al., 2024) | https://github.com/hvarfner/vanilla_bo_in_highdim | no licence file | re-implementations of the recipe in `src/spherevo/gp_competitors.py` and `src/spherevo/published_recipes.py` |
| Standard GP (Xu et al., 2025) | https://github.com/XZT008/Standard-GP-is-all-you-need-for-HDBO | no licence file | re-implementations in `src/spherevo/gp_competitors.py` and `src/spherevo/published_recipes.py` |
| MSR (Papenmeier et al., 2025) | https://github.com/lpapenme/understanding-hdbo-code | MIT | re-implementations in `src/spherevo/gp_competitors.py` and `src/spherevo/published_recipes.py` |
| COWBOYS (Moss et al., 2025) | https://github.com/henrymoss/ROTLSC | no licence file | an own, simplified solver (`src/spherevo/cowboys_solver.py`), which is not the published method |

No file of the three repositories without a licence is contained here. A check for verbatim reuse (code
lines of at least 40 characters, imports and comments excluded) found no own file that shares three or
more such lines with the Xu or Hvarfner repositories; it was repeated on 2026-10-23 for the files of the
published recipes. `src/spherevo/published_recipes.py` shares three such lines, all calls of library
functions, with the MIT-licensed Papenmeier repository.

## 3. Data and model weights

| asset | in this repository? | licence or terms | source |
|---|---|---|---|
| SELFIES Transformer VAE weights of LOL-BO, sha256 `edffc4b4c2d5582a6efa372eddae8913b8a38ba18d96006a6c3cdcdffacb31f0` | no; `scripts/download_weights.sh` | MIT (the licence of the LOL-BO repository, which contains the file) | https://github.com/nataliemaus/lolbo |
| ZINC-250k, the file `250k_rndm_zinc_drugs_clean_3.csv` of chemical_vae (Gómez-Bombarelli et al., 2018); molecules from the ZINC database (Irwin et al., 2012) | the first 10,000 molecules as `data/zinc_10k.txt`; the full file via `scripts/download_zinc250k.sh` | the file is distributed in an Apache-2.0 repository (`LICENSES/chemical_vae-Apache-2.0.txt`); the ZINC database itself states no formal licence | https://github.com/the-matter-lab/chemical_vae (formerly aspuru-guzik-group/chemical_vae), commit `a509dc61` |
| `results/precomputed/zinc_per_coord_bounds.json`, `zinc_latents_sample.npz`, `zinc_mean_norm.json` | yes | computed by the authors from ZINC-250k with the LOL-BO weights; as this repository | — |
| GuacaMol benchmark (oracles) | no; Python package `guacamol` 0.5.2 | MIT | https://github.com/BenevolentAI/guacamol |
| GuacaMol v1 training set (the corpus on which the LOL-BO authors trained the VAE; not read by any run here) | no | CC BY 4.0; derived from ChEMBL 24 (CC BY-SA 3.0) | https://doi.org/10.6084/m9.figshare.7322228.v2 |
| LassoBench | no; Python package at commit `cb9702e8` | MIT for the code and synthetic benchmarks; the real-world data sets under the terms of the LIBSVM website | https://github.com/ksehic/LassoBench |
| NAS-Bench-101 | no (the data file and the VAE weights of the stored runs are lost; the stored results are used as they are) | Apache-2.0 (repository) | https://github.com/google-research/nasbench |
| Chembed code and weights (cross-codec appendix; stored results only) | no | code: MIT (Copyright (C) 2025 Hugo Talibart); weights `3BioCompBio/chembed-default` on Hugging Face: no licence declared | https://github.com/3BioCompBio/chembed |

## 4. Python packages of the run environment

`environment/requirements-frozen.txt` pins the 114 packages of the run environment, and
`environment/PACKAGE_LICENCES.tsv` gives, for each of them, the version and the licence declared in its own
package metadata. None of them is redistributed here. The ones the code calls directly:

| package | version | licence |
|---|---|---|
| PyTorch | 2.11.0 (CUDA 13.0 build) | BSD-3-Clause |
| BoTorch | 0.17.2 | MIT |
| GPyTorch / linear-operator | 1.15.2 / 0.6.1 | MIT |
| NumPy / SciPy | 2.4.4 / 1.17.1 | BSD-3-Clause |
| RDKit | 2026.3.1 | BSD-3-Clause |
| selfies | 2.2.0 | Apache-2.0 |
| guacamol | 0.5.2 | MIT |
| pycma (`cma`) | 4.4.4 | BSD-3-Clause |
| pymanopt | 0.2.5 | BSD-3-Clause |
| Pyro (`pyro-ppl`) | 1.9.1 | Apache-2.0 |
| TensorFlow | 2.21.0 | Apache-2.0 |
| LassoBench | git `cb9702e8` | MIT (see section 3) |

The environment also contains the NVIDIA CUDA runtime libraries and the Intel MKL/OpenMP runtimes under
their vendors' proprietary redistribution licences (see the table file); they are installed from PyPI and are
not part of this repository.
