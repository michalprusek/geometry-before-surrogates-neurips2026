"""GuacaMol oracle for molecular scoring.

Provides a clean interface to score molecules on the 12 hard GuacaMol MPO
benchmark tasks. Wraps the guacamol library's standard benchmarks with
caching and batch scoring support.
"""

import logging
import math
from pathlib import Path
from typing import Optional

import numpy as np
from rdkit import Chem
from rdkit.Chem import Crippen
from rdkit.Chem.QED import qed

# Patch scipy for guacamol compatibility (scipy.histogram was removed in newer versions)
import scipy
if not hasattr(scipy, 'histogram'):
    scipy.histogram = np.histogram

logger = logging.getLogger(__name__)

# ── Task Constants ────────────────────────────────────────────────────────────

# All 12 hard GuacaMol MPO tasks
GUACAMOL_TASKS = [
    "med1",  # Median molecules 1 (camphor-menthol)
    "pdop",  # Perindopril MPO
    "adip",  # Amlodipine MPO
    "rano",  # Ranolazine MPO
    "osmb",  # Osimertinib MPO
    "siga",  # Sitagliptin MPO
    "zale",  # Zaleplon MPO
    "valt",  # Valsartan SMARTS
    "med2",  # Median molecules 2 (tadalafil-sildenafil)
    "dhop",  # Decoration Hop
    "shop",  # Scaffold Hop
    "fexo",  # Fexofenadine MPO
]

# PMO tasks available via guacamol (not in our 12 hard tasks)
PMO_EXTRA_TASKS = [
    "albuterol_similarity",
    "celecoxib_rediscovery",
    "mestranol_similarity",
    "thiothixene_rediscovery",
    "troglitazone_rediscovery",
    "isomers_c7h8n2o2",
    "isomers_c9h10n2o2pf2cl",
]

# Additional tasks (not used in paper benchmark)
EXTRA_TASKS = [
    "logp",  # Penalized logP
    "qed",   # Drug-likeness (QED)
]

GUACAMOL_TASK_DESCRIPTIONS = {
    "med1": "Median molecules 1: similarity to camphor and menthol",
    "pdop": "Perindopril MPO: ACE inhibitor optimization",
    "adip": "Amlodipine MPO: calcium channel blocker optimization",
    "rano": "Ranolazine MPO: antianginal optimization",
    "osmb": "Osimertinib MPO: EGFR inhibitor optimization",
    "siga": "Sitagliptin MPO: DPP-4 inhibitor optimization",
    "zale": "Zaleplon MPO: sedative-hypnotic optimization",
    "valt": "Valsartan SMARTS: ARB optimization with SMARTS constraint",
    "med2": "Median molecules 2: similarity to tadalafil and sildenafil",
    "dhop": "Decoration Hop: modify decorations while keeping scaffold",
    "shop": "Scaffold Hop: modify scaffold while keeping activity",
    "fexo": "Fexofenadine MPO: antihistamine optimization (less greasy)",
    "logp": "Penalized logP: lipophilicity with SA and ring penalty",
    "qed": "Quantitative Estimate of Drug-likeness",
}

# ── Benchmark Loading ─────────────────────────────────────────────────────────

# Lazy-loaded GuacaMol benchmark objects
_BENCHMARK_CACHE: dict = {}


def _get_benchmark(task_id: str):
    """Get or create GuacaMol benchmark object (lazy loading)."""
    if task_id in _BENCHMARK_CACHE:
        return _BENCHMARK_CACHE[task_id]

    from guacamol import standard_benchmarks

    benchmarks = {
        "med1": standard_benchmarks.median_camphor_menthol,
        "med2": standard_benchmarks.median_tadalafil_sildenafil,
        "pdop": standard_benchmarks.perindopril_rings,
        "osmb": standard_benchmarks.hard_osimertinib,
        "adip": standard_benchmarks.amlodipine_rings,
        "siga": standard_benchmarks.sitagliptin_replacement,
        "zale": standard_benchmarks.zaleplon_with_other_formula,
        "valt": standard_benchmarks.valsartan_smarts,
        "dhop": standard_benchmarks.decoration_hop,
        "shop": standard_benchmarks.scaffold_hop,
        "rano": standard_benchmarks.ranolazine_mpo,
        "fexo": standard_benchmarks.hard_fexofenadine,
    }

    # PMO extra tasks (similarity, isomers)
    pmo_benchmarks = {
        "albuterol_similarity": lambda: standard_benchmarks.similarity(
            smiles='CC(C)(C)c1ccc(cc1)C(O)C(=O)NCCc1ccc(O)cc1',
            name='Albuterol', fp_type='ECFP4', threshold=0.75),
        "celecoxib_rediscovery": lambda: standard_benchmarks.similarity(
            smiles='O=S(=O)(c3ccc(n1nc(cc1c2ccc(C)cc2)C(F)(F)F)cc3)N',
            name='Celecoxib', fp_type='ECFP4', threshold=1.0),
        "mestranol_similarity": lambda: standard_benchmarks.similarity(
            smiles='COc1ccc2[C@H]3CC[C@@]4(C)[C@@H](CC[C@@]4(O)C#C)[C@@H]3CCc2c1',
            name='Mestranol', fp_type='AP', threshold=0.75),
        "thiothixene_rediscovery": lambda: standard_benchmarks.similarity(
            smiles='C1=C(C=C2C(=C1)SC3=CC=CC=C3N2CCCO)S(=O)(=O)N(C)C',
            name='Thiothixene', fp_type='ECFP4', threshold=1.0),
        "troglitazone_rediscovery": lambda: standard_benchmarks.similarity(
            smiles='Cc1c(C)c2OC(C)(COc3ccc(CC4SC(=O)NC4=O)cc3)CCc2c(C)c1O',
            name='Troglitazone', fp_type='ECFP4', threshold=1.0),
        "isomers_c7h8n2o2": lambda: standard_benchmarks.isomers_c7h8n2o2(),
        "isomers_c9h10n2o2pf2cl": lambda: standard_benchmarks.isomers_c9h10n2o2pf2cl(),
    }

    all_benchmarks = {**benchmarks, **pmo_benchmarks}

    if task_id not in all_benchmarks:
        raise ValueError(
            f"Unknown task: {task_id}. "
            f"Available: {list(all_benchmarks.keys())} + {EXTRA_TASKS}"
        )

    _BENCHMARK_CACHE[task_id] = all_benchmarks[task_id]()
    return _BENCHMARK_CACHE[task_id]


# ── Scoring Functions ─────────────────────────────────────────────────────────

def is_valid_smiles(smiles: str) -> bool:
    """Check if SMILES string represents a valid molecule."""
    if smiles is None or len(smiles) == 0:
        return False
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None


def score_smiles(smiles: str, task_id: str) -> float:
    """Score a single SMILES string on a GuacaMol task.

    Args:
        smiles: SMILES string to score
        task_id: GuacaMol task identifier (e.g., "pdop", "siga")

    Returns:
        Score in [0, 1] for GuacaMol tasks, or task-specific range for logp/qed.
        Returns 0.0 for invalid molecules.
    """
    if smiles is None or len(smiles) == 0:
        return 0.0

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0.0

    try:
        if task_id == "logp":
            return _penalized_logp(mol)
        elif task_id == "qed":
            return qed(mol)
        elif task_id in GUACAMOL_TASKS or task_id in PMO_EXTRA_TASKS:
            benchmark = _get_benchmark(task_id)
            score = benchmark.objective.score(smiles)
            if score is None or score < 0:
                return 0.0
            return score
        else:
            raise ValueError(f"Unknown task: {task_id}")
    except Exception as e:
        logger.info(f"Scoring failed for '{smiles}' on task '{task_id}': {e}")
        return 0.0


def _penalized_logp(mol) -> float:
    """Calculate penalized logP (includes SA and ring penalties)."""
    try:
        from rdkit.Chem import RDConfig
        import os
        import sys
        sa_path = os.path.join(RDConfig.RDContribDir, "SA_Score")
        if sa_path not in sys.path:
            sys.path.insert(0, sa_path)
        import sascorer
    except ImportError:
        raise ImportError(
            "SA_Score (sascorer) not found. Install RDKit with contrib for penalized logP."
        )

    try:
        logp = Crippen.MolLogP(mol)
        sa = sascorer.calculateScore(mol)
        cycle_length = _cycle_score(mol)

        # Normalize using empirical means and stds
        score = (
            (logp - 2.45777691) / 1.43341767
            + (-sa + 3.05352042) / 0.83460587
            + (-cycle_length - -0.04861121) / 0.28746695
        )
        return score
    except Exception as e:
        logger.debug(f"Penalized logP calculation failed: {e}")
        return 0.0


def _cycle_score(mol) -> int:
    """Calculate cycle penalty for penalized logP."""
    import networkx as nx
    from rdkit.Chem import rdmolops

    cycle_list = nx.cycle_basis(nx.Graph(rdmolops.GetAdjacencyMatrix(mol)))
    if len(cycle_list) == 0:
        cycle_length = 0
    else:
        cycle_length = max([len(j) for j in cycle_list])
    if cycle_length <= 6:
        cycle_length = 0
    else:
        cycle_length = cycle_length - 6
    return cycle_length


def smiles_to_scores(smiles_list: list[str], task_id: str) -> np.ndarray:
    """Score a list of SMILES strings.

    Args:
        smiles_list: List of SMILES strings
        task_id: GuacaMol task identifier

    Returns:
        Array of scores [N]
    """
    scores = []
    for smiles in smiles_list:
        score = score_smiles(smiles, task_id)
        if score is not None and math.isfinite(score):
            scores.append(score)
        else:
            scores.append(0.0)
    return np.array(scores)


# ── Oracle Class ──────────────────────────────────────────────────────────────

class GuacaMolOracle:
    """Oracle for GuacaMol benchmark tasks.

    Provides a clean interface for scoring molecules, with caching and statistics.

    Example:
        >>> oracle = GuacaMolOracle(task_id="pdop")
        >>> score = oracle.score("CC(C)Cc1ccc(cc1)C(C)C(=O)O")  # Ibuprofen
        >>> print(f"Score: {score:.4f}")
    """

    def __init__(self, task_id: str = "pdop"):
        """Initialize oracle for a specific task.

        Args:
            task_id: GuacaMol task identifier. One of the 12 GUACAMOL_TASKS
                or EXTRA_TASKS ("logp", "qed").
        """
        all_valid = GUACAMOL_TASKS + PMO_EXTRA_TASKS + EXTRA_TASKS
        if task_id not in all_valid:
            raise ValueError(
                f"Unknown task: {task_id}. "
                f"Available: {all_valid}"
            )

        self.task_id = task_id
        self.num_calls = 0
        self._cache: dict[str, float] = {}

        # Pre-load benchmark to catch errors early
        if task_id in GUACAMOL_TASKS:
            _get_benchmark(task_id)

        logger.info(f"Initialized GuacaMolOracle for task '{task_id}'")

    def score(self, smiles: str, use_cache: bool = True) -> float:
        """Score a single SMILES string.

        Args:
            smiles: SMILES string to score
            use_cache: Whether to use cached scores

        Returns:
            Score value (0.0 for invalid molecules)
        """
        if use_cache and smiles in self._cache:
            return self._cache[smiles]

        score = score_smiles(smiles, self.task_id)
        self.num_calls += 1

        if use_cache:
            self._cache[smiles] = score

        return score

    def score_batch(
        self, smiles_list: list[str], use_cache: bool = True
    ) -> np.ndarray:
        """Score a batch of SMILES strings.

        Args:
            smiles_list: List of SMILES strings
            use_cache: Whether to use cached scores

        Returns:
            Array of scores [N]
        """
        scores = []
        for smiles in smiles_list:
            scores.append(self.score(smiles, use_cache=use_cache))
        return np.array(scores)

    def reset_cache(self) -> None:
        """Clear the score cache."""
        self._cache.clear()

    def reset_stats(self) -> None:
        """Reset call counter."""
        self.num_calls = 0

    @property
    def cache_size(self) -> int:
        """Number of cached scores."""
        return len(self._cache)

    def __repr__(self) -> str:
        return (
            f"GuacaMolOracle(task_id='{self.task_id}', "
            f"num_calls={self.num_calls}, cache_size={self.cache_size})"
        )


# ── Convenience Factory ──────────────────────────────────────────────────────

def create_oracle(task_id: str) -> GuacaMolOracle:
    """Create a GuacaMol oracle for the given task.

    Args:
        task_id: One of the 12 GUACAMOL_TASKS or EXTRA_TASKS.

    Returns:
        GuacaMolOracle instance (callable via .score() or .score_batch()).
    """
    return GuacaMolOracle(task_id=task_id)


# ── Data Loading ──────────────────────────────────────────────────────────────


def load_zinc_smiles(
    path: str = "data/zinc_10k.txt",
    n_samples: Optional[int] = None,
) -> list[str]:
    """Load SMILES from ZINC dataset.

    Args:
        path: Path to ZINC file (one SMILES per line).
            Defaults to ``data/zinc_10k.txt`` relative to project root.
        n_samples: Number of samples to load. None for all.

    Returns:
        List of SMILES strings
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"ZINC data file not found: {path}")

    logger.info(f"Loading ZINC data from {path}")

    with open(path, "r") as f:
        smiles_list = [line.strip() for line in f if line.strip()]

    if n_samples is not None and n_samples < len(smiles_list):
        smiles_list = smiles_list[:n_samples]

    logger.info(f"Loaded {len(smiles_list)} SMILES from ZINC")
    return smiles_list
