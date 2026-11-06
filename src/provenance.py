"""Collect provenance metadata for benchmark result traceability.

Every result JSON should include provenance so it's clear when, how,
and with what code it was generated.
"""

from __future__ import annotations

import importlib.metadata
import logging
import platform
import socket
import subprocess
import sys
from datetime import datetime


#: Seconds a git call may take. The camera-ready tooling treats an unknown
#: ``git_dirty`` as an invalid result, so a slow ``git status`` on a loaded
#: node must not time out; 120 s is the timeout of scripts/cr (was 5 s).
GIT_TIMEOUT_S = 120


def _git_output(*args: str) -> str | None:
    """Stripped stdout of a git command, or None if git failed or timed out."""
    try:
        r = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT_S,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def _run_git(*args: str) -> str:
    """Run a git command and return stripped stdout, or '' on failure."""
    out = _git_output(*args)
    return "" if out is None else out


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def collect_provenance() -> dict:
    """Return a dict of provenance metadata to embed in result JSONs.

    ``git_dirty`` is True or False when ``git status`` answered, and None when
    it failed or timed out: an unchecked tree is unknown, never "clean".
    """
    status = _git_output("status", "--porcelain")
    git_dirty = None if status is None else status != ""

    versions = {}
    for pkg in ("numpy", "torch", "cma", "scipy", "selfies", "poli", "poli-baselines"):
        v = _package_version(pkg)
        if v is not None:
            versions[pkg] = v

    cuda_version = None
    try:
        import torch
        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
    except Exception as e:
        logging.debug(f"Could not determine CUDA version: {type(e).__name__}: {e}")

    return {
        "provenance": {
            "git_commit": _run_git("rev-parse", "--short", "HEAD"),
            "git_branch": _run_git("rev-parse", "--abbrev-ref", "HEAD"),
            "git_dirty": git_dirty,
            "git_diff_stat": _run_git("diff", "--stat") if git_dirty else "",
            "cli_args": sys.argv,
            "hostname": socket.gethostname(),
            "python_version": platform.python_version(),
            "package_versions": versions,
            "cuda_version": cuda_version,
            "timestamp": datetime.now().isoformat(),
        }
    }
