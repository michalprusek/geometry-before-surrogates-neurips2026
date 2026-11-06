"""COWBOYS solver: Combinatorial Optimization With Bayesian Optimization on Your Structures.

Operates in fingerprint space rather than VAE latent space. At each step:
1. Generate N_candidates molecules by sampling VAE latent space.
2. Decode to SMILES and compute Morgan fingerprints.
3. Fit a GP with Tanimoto kernel on observed (fingerprint, score) pairs.
4. Score candidates via Thompson Sampling from the GP posterior.
5. Evaluate the top-1 candidate via the oracle.

This decouples the search representation (fingerprints) from the generative
model (VAE), which should provide better GP signal than fitting surrogates
directly in the high-D latent space.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
import gpytorch
from gpytorch.kernels import Kernel

from src.spherevo.solver import StepByStepSolver

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tanimoto (Jaccard) kernel for binary molecular fingerprints
# ---------------------------------------------------------------------------


class TanimotoKernel(Kernel):
    """Tanimoto (Jaccard) kernel for binary molecular fingerprints.

    k(x, y) = <x, y> / (||x||^2 + ||y||^2 - <x, y>)

    For binary fingerprints, this equals |intersection| / |union|.
    Operates on dense float tensors (0/1 valued for true fingerprints,
    but works with any non-negative features).
    """

    has_lengthscale = False

    def forward(self, x1, x2, diag=False, **params):
        # x1: [n, d], x2: [m, d]
        dot = x1 @ x2.T  # [n, m]
        x1_sq = (x1 ** 2).sum(dim=-1, keepdim=True)  # [n, 1]
        x2_sq = (x2 ** 2).sum(dim=-1, keepdim=True).T  # [1, m]
        denom = x1_sq + x2_sq - dot
        K = dot / (denom + 1e-8)
        if diag:
            return K.diag()
        return K


# ---------------------------------------------------------------------------
# ExactGP model with Tanimoto kernel
# ---------------------------------------------------------------------------


class _TanimotoGP(gpytorch.models.ExactGP):
    """ExactGP with ScaleKernel(TanimotoKernel) + constant mean."""

    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(TanimotoKernel())

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)


# ---------------------------------------------------------------------------
# COWBOYS Solver
# ---------------------------------------------------------------------------


class COWBOYSSolver(StepByStepSolver):
    """Combinatorial Optimization With Bayesian Optimization on Your Structures.

    Uses a GP with Tanimoto kernel in Morgan fingerprint space for surrogate
    modelling, while generating candidates by sampling the VAE latent space.

    Parameters
    ----------
    black_box : LatentBlackBox
        Must expose ``decode_fn(z) -> list[str]`` for decoding latent vectors
        to SMILES strings.
    x0 : np.ndarray
        Cold-start latent vectors, shape ``[N, D]``.
    y0 : np.ndarray
        Cold-start scores, shape ``[N, 1]``.
    codec : object
        VAE codec (unused directly, but kept for interface symmetry with other
        solvers that receive it from the factory).
    n_candidates : int
        Number of candidate molecules to generate per step.
    fp_radius : int
        Morgan fingerprint radius.
    fp_nbits : int
        Morgan fingerprint bit length.
    """

    def __init__(
        self,
        black_box: Any,
        x0: np.ndarray,
        y0: np.ndarray,
        *,
        codec: Any = None,
        n_candidates: int = 100,
        fp_radius: int = 2,
        fp_nbits: int = 2048,
    ) -> None:
        super().__init__(black_box, x0, y0)
        self._codec = codec
        self.n_candidates = n_candidates
        self.fp_radius = fp_radius
        self.fp_nbits = fp_nbits

        # GP training data: (fingerprint_tensor, score)
        self._fp_X: list[torch.Tensor] = []  # each [fp_nbits]
        self._fp_y: list[float] = []

        # Cap decoder max generation length to avoid O(N³) blowup for random
        # z-vectors that never produce a stop token. Drug-like molecules are
        # typically <100 SELFIES tokens; 128 is more than sufficient.
        self._orig_max_string_length: int | None = None
        if self._codec is not None and hasattr(self._codec, "model"):
            self._orig_max_string_length = self._codec.model.max_string_length
            self._codec.model.max_string_length = 80

        # Initialize fingerprint data from cold-start molecules
        self._init_fingerprints()

    # -- Fingerprint utilities ------------------------------------------------

    def _smiles_to_fingerprint(self, smiles: str) -> torch.Tensor | None:
        """Convert SMILES string to Morgan fingerprint tensor.

        Returns None if SMILES is invalid or RDKit cannot parse it.
        """
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem
        except ImportError:
            logger.error("RDKit is required for COWBOYSSolver")
            raise

        if not smiles:
            return None

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol, self.fp_radius, nBits=self.fp_nbits
        )
        return torch.tensor(list(fp), dtype=torch.float64)

    def _init_fingerprints(self) -> None:
        """Decode cold-start latent vectors to SMILES and compute fingerprints."""
        smiles_list = self._black_box.decode_fn(self._x0)
        n_valid = 0
        for smi, score in zip(smiles_list, self._y0.flatten()):
            if not smi or np.isnan(score):
                continue
            fp = self._smiles_to_fingerprint(smi)
            if fp is not None:
                self._fp_X.append(fp)
                self._fp_y.append(float(score))
                n_valid += 1
        logger.info(
            f"COWBOYS: initialized with {n_valid}/{len(smiles_list)} valid "
            f"fingerprints from cold start"
        )

    # -- GP fitting -----------------------------------------------------------

    def _fit_gp(self) -> _TanimotoGP:
        """Fit ExactGP with Tanimoto kernel on accumulated fingerprint data.

        Returns the fitted model in eval mode.

        Raises RuntimeError if fitting fails (caller should handle).
        """
        train_x = torch.stack(self._fp_X).double()
        train_y = torch.tensor(self._fp_y, dtype=torch.float64)

        # Standardize targets for numerical stability
        self._y_mean = train_y.mean()
        self._y_std = train_y.std().clamp(min=1e-6)
        train_y_norm = (train_y - self._y_mean) / self._y_std

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        likelihood.double()
        model = _TanimotoGP(train_x, train_y_norm, likelihood)
        model.double()

        model.train()
        likelihood.train()

        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

        # Short training (fingerprint GPs converge quickly)
        model.train()
        for _ in range(50):
            optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, train_y_norm)
            loss.backward()
            optimizer.step()

        model.eval()
        likelihood.eval()
        return model

    # -- Candidate generation -------------------------------------------------

    def next_candidate(self) -> np.ndarray:
        """Generate candidates via VAE, score via fingerprint GP, return best latent.

        1. Sample random latent vectors near the cold-start distribution.
        2. Decode to SMILES via the black_box's decode function.
        3. Compute Morgan fingerprints for valid molecules.
        4. If GP data is available, score via Thompson Sampling from GP posterior.
        5. Return the latent vector of the highest-scoring candidate.

        Falls back to random selection if GP fitting fails or no valid
        fingerprints are available.
        """
        # 1. Generate random latent vectors (near cold-start distribution)
        center = self._x0.mean(axis=0)
        z_candidates = (
            center + np.random.randn(self.n_candidates, self._dim) * self.mean_norm * 0.5
        )

        # 2. Decode to SMILES
        smiles_list = self._black_box.decode_fn(z_candidates)

        # 3. Compute fingerprints for valid molecules
        fps: list[torch.Tensor] = []
        valid_indices: list[int] = []
        for i, smi in enumerate(smiles_list):
            if smi:
                fp = self._smiles_to_fingerprint(smi)
                if fp is not None:
                    fps.append(fp)
                    valid_indices.append(i)

        if not fps:
            # No valid molecules decoded -- return random cold-start vector
            logger.warning("COWBOYS: no valid fingerprints from candidates, using random fallback")
            idx = np.random.randint(len(self._x0))
            return self._x0[idx].reshape(1, -1)

        # 4. Score via GP posterior (Thompson Sampling)
        if len(self._fp_X) >= 2:
            try:
                model = self._fit_gp()
                fps_tensor = torch.stack(fps).double()
                with torch.no_grad(), gpytorch.settings.fast_pred_var():
                    posterior = model(fps_tensor)
                    # Thompson sample: draw one sample from the posterior
                    sample = posterior.rsample(torch.Size([1]))  # [1, n_valid]
                    best_idx = sample.squeeze().argmax().item()
            except Exception as e:
                logger.warning(f"COWBOYS: GP fitting/prediction failed ({e}), using random fallback")
                best_idx = np.random.randint(len(valid_indices))
        else:
            # Not enough data for GP -- pick random valid candidate
            best_idx = np.random.randint(len(valid_indices))

        # 5. Return the latent vector of the best candidate
        return z_candidates[valid_indices[best_idx]].reshape(1, -1)

    # -- Step override --------------------------------------------------------

    def step(self) -> float:
        """Run one optimization iteration with fingerprint bookkeeping.

        Extends the base step() to also record the fingerprint of the
        evaluated molecule for future GP training.
        """
        z = self.next_candidate()  # [1, D]
        scores = self._black_box(z)  # [1, 1]
        score = float(np.asarray(scores).flatten()[0])

        # Update history (same as StepByStepSolver.step)
        y = np.array([[score]])
        self.history["x"].append(z.reshape(1, -1))
        self.history["y"].append(y)
        self._all_scores.append(score)

        self.post_update(z, y)
        self.iteration += 1

        # Record fingerprint for GP training data
        if not np.isnan(score):
            self._record_evaluation(z, score)

        return score

    def _record_evaluation(self, z: np.ndarray, score: float) -> None:
        """Decode evaluated latent vector and add its fingerprint to GP data."""
        try:
            smiles_list = self._black_box.decode_fn(z.reshape(1, -1))
            if smiles_list and smiles_list[0]:
                fp = self._smiles_to_fingerprint(smiles_list[0])
                if fp is not None:
                    self._fp_X.append(fp)
                    self._fp_y.append(score)
        except Exception as e:
            logger.warning(f"COWBOYS: failed to record fingerprint: {e}")
