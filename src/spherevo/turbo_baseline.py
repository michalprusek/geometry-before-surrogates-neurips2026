from __future__ import annotations

"""TuRBO baseline for comparison with Spherical Subspace BO.

TuRBO operates a GP in the full 256D latent space with a local trust region,
whereas Subspace BO projects to S^15 for a tractable GP.

Follows the original TuRBO paper (Eriksson et al., 2019):
- Thompson Sampling acquisition (not EI)
- [0,1]^d min-max normalization
- GP hyperparameter constraints matching BoTorch tutorial
- Lengthscale-weighted trust region clamped to [0,1]^d

Usage:
    CUDA_VISIBLE_DEVICES=0 python -m src.spherevo.turbo_baseline \
        --task-id pdop --n-cold-start 100 --iterations 500 --seed 42
"""

import argparse
import json
import logging
import math
import os
import random
from datetime import datetime

import numpy as np
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.generation import MaxPosteriorSampling
from botorch.models import SingleTaskGP
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.quasirandom import SobolEngine

from src.spherevo.gp_diagnostics import GPDiagnostics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class TurboState:
    """Trust region state for TuRBO."""

    def __init__(
        self,
        dim: int,
        batch_size: int = 1,
        length: float = 0.8,
        length_min: float = 0.5**7,
        length_max: float = 1.6,
        failure_counter: int = 0,
        failure_tolerance: int | None = None,
        success_counter: int = 0,
        success_tolerance: int = 3,
    ):
        self.dim = dim
        self.batch_size = batch_size
        self.length = length
        self.length_min = length_min
        self.length_max = length_max
        self.failure_counter = failure_counter
        # Paper formula: ceil(max(4/b, d/b)) -- Eriksson et al. 2019
        self.failure_tolerance = (
            failure_tolerance if failure_tolerance is not None
            else math.ceil(max(4.0 / batch_size, dim / batch_size))
        )
        self.success_counter = success_counter
        self.success_tolerance = success_tolerance
        self.best_value = -float("inf")
        self.restart_triggered = False

    def update(self, y_next: float):
        """Update state based on new observation (Eriksson et al. 2019, Algorithm 1)."""
        if y_next > self.best_value + 1e-3 * abs(self.best_value):
            self.success_counter += 1
            self.failure_counter = 0
        else:
            self.success_counter = 0
            self.failure_counter += 1

        if self.success_counter == self.success_tolerance:
            self.length = min(2.0 * self.length, self.length_max)
            self.success_counter = 0
        elif self.failure_counter == self.failure_tolerance:
            self.length = self.length / 2.0
            self.failure_counter = 0

        self.best_value = max(self.best_value, y_next)

        if self.length < self.length_min:
            self.restart_triggered = True


class TuRBOBaseline:
    """TuRBO baseline operating in full 256D latent space.

    Follows Eriksson et al. (2019) with Thompson Sampling acquisition,
    [0,1]^d min-max normalization, and constrained GP hyperparameters.
    """

    def __init__(
        self,
        codec,
        oracle,
        input_dim: int = 256,
        device: str = "cuda",
        n_candidates: int = 5000,
        trust_region: float = 0.8,
        seed: int = 42,
        verbose: bool = True,
        failure_tolerance=None,
    ):
        self.codec = codec
        self.oracle = oracle
        self.input_dim = input_dim
        self.device = device
        self.n_candidates = n_candidates
        self.trust_region = trust_region
        self.seed = seed
        self.verbose = verbose

        self.train_Z = None
        self.train_Y = None
        self.smiles_observed = set()
        self.best_score = -float("inf")
        self.best_smiles = None
        self.best_z = None
        self.mean_norm = None
        self.history = []
        self.gp = None
        self.turbo_state = None
        self._failure_tolerance_override = failure_tolerance
        self.gp_diagnostics = GPDiagnostics(verbose=True)
        self.diagnostic_history = []

    def cold_start(self, smiles_list: list, scores: list):
        """Initialize with cold start data."""
        logger.info(f"Cold start with {len(smiles_list)} molecules")

        z_list = []
        valid_scores = []
        valid_smiles = []

        for smi, score in zip(smiles_list, scores):
            try:
                z = self.codec.encode([smi])
                if z is not None and not torch.isnan(z).any():
                    z_list.append(z)
                    valid_scores.append(score)
                    valid_smiles.append(smi)
                    self.smiles_observed.add(smi)
            except (ValueError, RuntimeError) as e:
                logger.info(f"Failed to encode {smi[:50]}...: {e}")
                continue
            except (torch.cuda.OutOfMemoryError, MemoryError):
                raise

        if not z_list:
            raise ValueError("No valid molecules in cold start")

        self.train_Z = torch.cat(z_list, dim=0).to(self.device)
        self.train_Y = torch.tensor(valid_scores, dtype=torch.float32, device=self.device)

        self.mean_norm = self.train_Z.norm(dim=-1).mean().item()

        best_idx = self.train_Y.argmax()
        self.best_score = self.train_Y[best_idx].item()
        self.best_smiles = valid_smiles[best_idx]
        self.best_z = self.train_Z[best_idx].clone()

        self.turbo_state = TurboState(
            dim=self.input_dim,
            length=self.trust_region,
            failure_tolerance=self._failure_tolerance_override,
        )
        self.turbo_state.best_value = self.best_score

        logger.info(
            f"Cold start: {len(valid_smiles)} molecules, best={self.best_score:.4f}, "
            f"failure_tol={self.turbo_state.failure_tolerance}"
        )
        logger.info(f"Mean norm: {self.mean_norm:.2f}")

        self.history.append({
            "iteration": 0,
            "best_score": self.best_score,
            "n_evaluated": len(self.smiles_observed),
        })

    def _normalize_z(self, z: torch.Tensor) -> torch.Tensor:
        """Normalize latent vectors to [0,1]^d via min-max (per TuRBO paper)."""
        self._z_lb = self.train_Z.min(dim=0).values
        self._z_ub = self.train_Z.max(dim=0).values
        return (z - self._z_lb) / (self._z_ub - self._z_lb + 1e-8)

    def _denormalize_z(self, z_norm: torch.Tensor) -> torch.Tensor:
        """Denormalize latent vectors from [0,1]^d back to original space."""
        return z_norm * (self._z_ub - self._z_lb + 1e-8) + self._z_lb

    def _fit_gp(self):
        """Fit GP in [0,1]^d normalized latent space with constrained hyperparameters.

        Constraints follow the BoTorch TuRBO tutorial (Eriksson et al. 2019):
        - Noise: [1e-8, 1e-3]
        - Lengthscale: [0.005, 4.0]
        - Outputscale: [0.05, 20.0]
        """
        train_Z_norm = self._normalize_z(self.train_Z)

        # Check for NaN/Inf in normalized data
        if torch.isnan(train_Z_norm).any() or torch.isinf(train_Z_norm).any():
            logger.warning("NaN/Inf in normalized training data, clamping")
            train_Z_norm = torch.nan_to_num(train_Z_norm, nan=0.0, posinf=1.0, neginf=0.0)

        y_mean = self.train_Y.mean()
        y_std = self.train_Y.std().clamp(min=1e-6)
        train_Y_norm = (self.train_Y - y_mean) / y_std

        if torch.isnan(train_Y_norm).any():
            logger.warning("NaN in normalized Y data, clamping")
            train_Y_norm = torch.nan_to_num(train_Y_norm, nan=0.0)

        prev_gp = self.gp  # Save properly fitted GP from previous iteration
        try:
            likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
            covar_module = ScaleKernel(
                MaternKernel(
                    nu=2.5,
                    ard_num_dims=self.input_dim,
                    lengthscale_constraint=Interval(0.005, 4.0),
                ),
                outputscale_constraint=Interval(0.05, 20.0),
            )
            self.gp = SingleTaskGP(
                train_Z_norm.double(),
                train_Y_norm.double().unsqueeze(-1),
                likelihood=likelihood,
                covar_module=covar_module,
            )

            mll = ExactMarginalLogLikelihood(self.gp.likelihood, self.gp)
            fit_gpytorch_mll(mll)
        except (RuntimeError, torch.linalg.LinAlgError) as e:
            if isinstance(e, (torch.cuda.OutOfMemoryError, MemoryError)):
                raise
            logger.error(f"TuRBO GP fit failed ({type(e).__name__}): {e}")
            try:
                # Fallback with wider noise constraint
                fallback_likelihood = GaussianLikelihood(noise_constraint=Interval(1e-4, 1.0))
                fallback_covar = ScaleKernel(
                    MaternKernel(
                        nu=2.5,
                        ard_num_dims=self.input_dim,
                        lengthscale_constraint=Interval(0.005, 4.0),
                    ),
                    outputscale_constraint=Interval(0.05, 20.0),
                )
                self.gp = SingleTaskGP(
                    train_Z_norm.double(),
                    train_Y_norm.double().unsqueeze(-1),
                    likelihood=fallback_likelihood,
                    covar_module=fallback_covar,
                )
                self.gp.likelihood.noise = 0.1
            except (RuntimeError, torch.linalg.LinAlgError) as e2:
                if isinstance(e2, (torch.cuda.OutOfMemoryError, MemoryError)):
                    raise
                logger.error(f"TuRBO fallback GP also failed ({type(e2).__name__}): {e2}")
                self.gp = prev_gp  # Restore actual previous GP
                if self.gp is None:
                    raise RuntimeError("No GP available") from e2
                logger.warning("Restored previous iteration's GP as last-resort fallback")

        self.gp.eval()
        self._y_mean = y_mean
        self._y_std = y_std
        self._train_Z_norm = train_Z_norm  # Store for diagnostics

    def _generate_candidates(self) -> torch.Tensor:
        """Generate candidates in lengthscale-weighted TR with probabilistic perturbation.

        Follows TuRBO paper (Eriksson et al. 2019, Section 3):
        - TR sides scaled by learned GP lengthscales
        - Each dimension perturbed with probability min(20/d, 1)
        - TR bounds clamped to [0,1]^d (min-max normalized space)
        """
        z_center_norm = self._normalize_z(self.best_z.unsqueeze(0)).squeeze(0)

        # Lengthscale-weighted trust region bounds
        ls = self.gp.covar_module.base_kernel.lengthscale.detach().float().view(-1)
        weights = ls / ls.mean()
        weights = weights / torch.prod(weights.pow(1.0 / len(weights)))

        tr_lb = torch.clamp(z_center_norm - weights * self.turbo_state.length / 2.0, 0.0, 1.0)
        tr_ub = torch.clamp(z_center_norm + weights * self.turbo_state.length / 2.0, 0.0, 1.0)

        sobol = SobolEngine(self.input_dim, scramble=True, seed=self.seed + len(self.history))
        pert = sobol.draw(self.n_candidates).to(dtype=torch.float32, device=self.device)
        pert = tr_lb + (tr_ub - tr_lb) * pert

        # Probabilistic perturbation mask (per TuRBO paper)
        prob_perturb = min(20.0 / self.input_dim, 1.0)
        mask = torch.rand(self.n_candidates, self.input_dim, device=self.device) <= prob_perturb
        ind = torch.where(mask.sum(dim=1) == 0)[0]
        if len(ind) > 0:
            mask[ind, torch.randint(0, self.input_dim, size=(len(ind),), device=self.device)] = True

        X_cand = z_center_norm.expand(self.n_candidates, -1).clone()
        X_cand[mask] = pert[mask]

        return X_cand

    def _select_candidate(self, z_cand_norm: torch.Tensor) -> torch.Tensor:
        """Select best candidate using Thompson Sampling (per TuRBO paper)."""
        ts = MaxPosteriorSampling(model=self.gp, replacement=False)
        with torch.no_grad():
            X_next = ts(z_cand_norm.double(), num_samples=1)
        return X_next.squeeze(0).float()

    def _sanitize_training_data(self) -> bool:
        """Remove any NaN entries from training data. Returns True if data was cleaned."""
        nan_mask_z = torch.isnan(self.train_Z).any(dim=1)
        nan_mask_y = torch.isnan(self.train_Y)
        nan_mask = nan_mask_z | nan_mask_y
        if nan_mask.any():
            n_nan = nan_mask.sum().item()
            logger.warning(f"Removing {n_nan} NaN entries from training data")
            keep = ~nan_mask
            self.train_Z = self.train_Z[keep]
            self.train_Y = self.train_Y[keep]
            return True
        return False

    def _step(self, iteration: int = 0) -> tuple[float | None, str]:
        """Single optimization step. Returns (score, smiles); score is None if duplicate/decode failed."""
        # Sanitize training data before GP fitting to prevent NaN cascade
        self._sanitize_training_data()

        self._fit_gp()

        z_cand_norm = self._generate_candidates()

        # Check for NaN in candidates (from bad GP lengthscales)
        nan_cand = torch.isnan(z_cand_norm).any(dim=1)
        if nan_cand.all():
            logger.warning(f"Iter {iteration}: ALL candidates are NaN, skipping step")
            return None, ""
        if nan_cand.any():
            z_cand_norm = z_cand_norm[~nan_cand]

        if iteration % 10 == 0:
            metrics = self.gp_diagnostics.analyze(
                self.gp,
                self._train_Z_norm,
                (self.train_Y - self._y_mean) / self._y_std,
                z_cand_norm[:100],  # Sample of candidates for extrapolation check
            )
            self.gp_diagnostics.log_summary(metrics, prefix=f"[Iter {iteration}]")
            self.diagnostic_history.append(self.gp_diagnostics.get_summary_dict(metrics))

        z_opt_norm = self._select_candidate(z_cand_norm)
        z_opt = self._denormalize_z(z_opt_norm.unsqueeze(0)).float()

        # Check for NaN in selected candidate before adding to training data
        if torch.isnan(z_opt).any():
            logger.warning(f"Iter {iteration}: selected candidate has NaN, skipping step")
            return None, ""

        try:
            smiles = self.codec.decode(z_opt)[0]
        except (ValueError, RuntimeError, IndexError) as e:
            # Expected decoding failures
            logger.info(f"Decode failed: {e}")
            return None, ""
        except (torch.cuda.OutOfMemoryError, MemoryError):
            # Critical memory errors - re-raise
            raise

        if smiles in self.smiles_observed:
            return None, smiles

        score = self.oracle.score(smiles)
        self.smiles_observed.add(smiles)

        self.train_Z = torch.cat([self.train_Z, z_opt], dim=0)
        self.train_Y = torch.cat([self.train_Y, torch.tensor([score], device=self.device, dtype=torch.float32)])

        if score > self.best_score:
            self.best_score = score
            self.best_smiles = smiles
            self.best_z = z_opt.squeeze()

        self.turbo_state.update(score)

        return score, smiles

    def optimize(self, n_iterations: int, log_interval: int = 10):
        """Run optimization loop."""
        logger.info(f"Starting TuRBO optimization for {n_iterations} iterations")

        for i in range(1, n_iterations + 1):
            score, _smiles = self._step(iteration=i)

            if self.turbo_state.restart_triggered:
                logger.info(f"Iter {i}: Trust region restart triggered")
                self.turbo_state = TurboState(
                    dim=self.input_dim,
                    length=self.trust_region,
                    failure_tolerance=self._failure_tolerance_override,
                )
                self.turbo_state.best_value = self.best_score

            if i % log_interval == 0 or i == n_iterations:
                logger.info(
                    f"Iter {i}/{n_iterations}: best={self.best_score:.4f}, "
                    f"n_eval={len(self.smiles_observed)}, "
                    f"tr_len={self.turbo_state.length:.4f}"
                )

            self.history.append({
                "iteration": i,
                "best_score": self.best_score,
                "n_evaluated": len(self.smiles_observed),
            })

        logger.info(f"Optimization complete: best={self.best_score:.4f}")
        logger.info(f"Best SMILES: {self.best_smiles}")


def main():
    parser = argparse.ArgumentParser(description="TuRBO Baseline on GuacaMol")

    parser.add_argument("--task-id", type=str, default="pdop")
    parser.add_argument("--n-cold-start", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--n-candidates", type=int, default=5000)
    parser.add_argument("--trust-region", type=float, default=0.8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    logger.info("Loading codec and oracle...")
    from src.codec import SELFIESVAECodec
    from src.oracle.guacamol import GuacaMolOracle

    codec = SELFIESVAECodec.from_pretrained(device=args.device)
    oracle = GuacaMolOracle(task_id=args.task_id)

    logger.info("Loading cold start data...")
    from src.oracle.guacamol import load_zinc_smiles

    zinc_smiles = load_zinc_smiles(n_samples=10000)
    rng = np.random.RandomState(args.seed)
    cs_indices = rng.choice(len(zinc_smiles), args.n_cold_start, replace=False)
    smiles_list = [zinc_smiles[i] for i in cs_indices]
    scores_list = [oracle.score(smi) for smi in smiles_list]

    optimizer = TuRBOBaseline(
        codec=codec,
        oracle=oracle,
        input_dim=256,
        device=args.device,
        n_candidates=args.n_candidates,
        trust_region=args.trust_region,
        seed=args.seed,
    )

    optimizer.cold_start(smiles_list, scores_list)
    optimizer.optimize(n_iterations=args.iterations, log_interval=10)

    results_dir = "results/guacamol"
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(results_dir, f"turbo_{args.task_id}_seed{args.seed}_{timestamp}.json")

    results = {
        "method": "turbo",
        "task_id": args.task_id,
        "best_score": optimizer.best_score,
        "best_smiles": optimizer.best_smiles,
        "n_evaluated": len(optimizer.smiles_observed),
        "history": optimizer.history,
        "gp_diagnostics": optimizer.diagnostic_history,
        "args": vars(args),
    }

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
