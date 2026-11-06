"""Unified result loading, deduplication, and filtering.

Centralizes result I/O patterns shared across analysis scripts.
Prefer this module over custom load/dedup functions.
"""

from __future__ import annotations

import json
import os
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.analysis.config import (
    BENCHMARK_SEEDS,
    BENCHMARK_TASKS,
    RESULT_DIRS,
    SEED_REGEX,
)


def load_json_result(path: str | Path) -> dict:
    """Load a single JSON result file."""
    with open(path) as f:
        return json.load(f)


def get_best_score(data: dict) -> float:
    """Extract best score from a result dict (handles both formats)."""
    history = data.get("history", {})
    best_scores = history.get("best_score", [])
    if best_scores:
        return max(best_scores)
    return data.get("best_score", float("nan"))


def get_n_evaluated(data: dict) -> int:
    """Extract number of evaluations from a result dict."""
    return data.get("n_evaluated", 0)


def get_n_iterations(data: dict) -> int:
    """Extract iteration count from history."""
    return len(data.get("history", {}).get("best_score", []))


def get_surrogate_fallback_count(data: dict) -> int:
    """Number of surrogate-fallback events recorded by the solver.

    Returns 0 for methods that do not expose this field (e.g. random search,
    GeoWalk, classical poli baselines). The GP-based competitors
    (Hvarfner / Papenmeier / Xu / Tripp / RF, sphere and box variants)
    persist this via ``run_benchmark.py``.
    """
    return int(data.get("surrogate_fallback_count", 0))


def find_method_results(
    result_dir: str | Path,
    tasks: list[str] | None = None,
    seeds: list[int] | None = None,
    min_n_evaluated: int | None = None,
    max_n_evaluated: int | None = None,
    max_fallback_count: int | None = 0,
) -> dict[tuple[str, int], dict]:
    """Load results from a method directory with deduplication by (task, seed).

    Uses earliest mtime for deduplication when multiple files match.

    ``max_fallback_count`` (default 0) excludes runs whose solver hit a
    surrogate fallback during optimization. The GP competitors document
    that "runs with nonzero fallback counts must be excluded from aggregate
    statistics" because a fallback step injects a random proposal into the
    trajectory and contaminates the surrogate's evaluation. Set to ``None``
    to disable the filter.

    Args:
        result_dir: Directory containing JSON result files for one method.
        tasks: Filter to these tasks (default: BENCHMARK_TASKS).
        seeds: Filter to these seeds (default: BENCHMARK_SEEDS).
        min_n_evaluated: Minimum evaluations to include.
        max_n_evaluated: Maximum evaluations to include.

    Returns:
        Dict mapping (task, seed) -> result data dict.
    """
    result_dir = Path(result_dir)
    if not result_dir.exists():
        return {}

    if tasks is None:
        tasks = BENCHMARK_TASKS
    if seeds is None:
        seeds = BENCHMARK_SEEDS

    # Collect candidates grouped by (task, seed) for dedup
    candidates: dict[tuple[str, int], list[tuple[float, Path]]] = defaultdict(list)

    for fpath in result_dir.glob("*.json"):
        fname = fpath.name
        if fname.startswith("gpu"):  # skip log files
            continue

        try:
            data = load_json_result(fpath)
        except (json.JSONDecodeError, OSError) as e:
            warnings.warn(f"Skipping unreadable result {fpath}: {e}")
            continue

        # Extract task and seed from data or filename
        task = data.get("task_id")
        seed = data.get("seed")

        # Fallback: parse seed from filename
        if seed is None:
            seed_match = SEED_REGEX.search(fname)
            if seed_match:
                seed = int(seed_match.group(1))

        if task is None or seed is None:
            continue
        if task not in tasks or seed not in seeds:
            continue

        candidates[(task, seed)].append((os.path.getmtime(str(fpath)), fpath))

    # Dedup: earliest mtime wins
    results = {}
    for key, entries in candidates.items():
        entries.sort(key=lambda x: x[0])  # sort by mtime ascending
        fpath = entries[0][1]

        try:
            data = load_json_result(fpath)
        except (json.JSONDecodeError, OSError) as e:
            warnings.warn(f"Skipping unreadable result {fpath}: {e}")
            continue

        # Apply filters
        if min_n_evaluated is not None and get_n_evaluated(data) < min_n_evaluated:
            continue
        if max_n_evaluated is not None and get_n_evaluated(data) > max_n_evaluated:
            continue
        if (
            max_fallback_count is not None
            and get_surrogate_fallback_count(data) > max_fallback_count
        ):
            warnings.warn(
                f"Excluding {fpath.name}: "
                f"surrogate_fallback_count={get_surrogate_fallback_count(data)} "
                f"> max={max_fallback_count} "
                f"(set max_fallback_count=None to override)"
            )
            continue

        results[key] = data

    return results


def find_all_method_results(
    base_dir: str | Path | None = None,
    methods: list[str] | None = None,
    tasks: list[str] | None = None,
    seeds: list[int] | None = None,
) -> dict[str, dict[tuple[str, int], dict]]:
    """Load results for all methods under the guacamol results directory.

    Args:
        base_dir: Base results directory (default: RESULT_DIRS["guacamol"]).
        methods: Filter to these method subdirectory names (default: all).
        tasks: Filter to these tasks (default: BENCHMARK_TASKS).
        seeds: Filter to these seeds (default: BENCHMARK_SEEDS).

    Returns:
        Dict mapping method_name -> {(task, seed): result_data}.
    """
    if base_dir is None:
        base_dir = Path(RESULT_DIRS["guacamol"])
    else:
        base_dir = Path(base_dir)

    if not base_dir.exists():
        return {}

    results: dict[str, dict[tuple[str, int], dict]] = {}

    for subdir in sorted(base_dir.iterdir()):
        if not subdir.is_dir():
            continue
        method_name = subdir.name
        if methods and method_name not in methods:
            continue

        method_results = find_method_results(subdir, tasks=tasks, seeds=seeds)
        if method_results:
            results[method_name] = method_results

    return results


def compute_task_means(
    results: dict[tuple[str, int], dict],
    tasks: list[str] | None = None,
    seeds: list[int] | None = None,
) -> dict[str, float]:
    """Compute per-task mean scores from (task, seed) -> data results.

    Returns:
        Dict mapping task -> mean score.
    """
    if tasks is None:
        tasks = BENCHMARK_TASKS
    if seeds is None:
        seeds = BENCHMARK_SEEDS

    task_means = {}
    for task in tasks:
        scores = []
        for seed in seeds:
            data = results.get((task, seed))
            if data is not None:
                scores.append(get_best_score(data))
        if scores:
            task_means[task] = float(np.mean(scores))

    return task_means


def compute_cross_task_mean(
    results: dict[tuple[str, int], dict],
    tasks: list[str] | None = None,
    seeds: list[int] | None = None,
) -> float:
    """Compute cross-task grand mean (mean of task means)."""
    task_means = compute_task_means(results, tasks, seeds)
    if task_means:
        return float(np.mean(list(task_means.values())))
    return float("nan")
