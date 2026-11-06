"""``src/provenance.py``: an unchecked tree is unknown, never clean.

``git_dirty`` is True or False only when ``git status`` answered; when git
fails or times out it is None, and the camera-ready queue refuses such a file
(``validate_result`` accepts only ``git_dirty is False``).
"""
from __future__ import annotations

import subprocess

import pytest

from scripts.cr import cr_queue
from src import provenance

SHA = "0123456789abcdef0123456789abcdef01234567"


def _fake_git(status_behaviour):
    """subprocess.run stand-in: ``git status`` behaves as asked, the rest succeeds."""

    def run(cmd, **kwargs):
        args = cmd[1:]
        if args[:1] == ["status"]:
            if status_behaviour == "timeout":
                raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout"))
            if status_behaviour == "oserror":
                raise OSError("no git")
            if status_behaviour == "fails":
                return subprocess.CompletedProcess(cmd, 128, "", "fatal: not a git repository")
            return subprocess.CompletedProcess(cmd, 0, status_behaviour, "")
        if args[:2] == ["rev-parse", "--short"]:
            return subprocess.CompletedProcess(cmd, 0, SHA[:7] + "\n", "")
        if args[:2] == ["rev-parse", "--abbrev-ref"]:
            return subprocess.CompletedProcess(cmd, 0, "camera-ready\n", "")
        if args[:1] == ["diff"]:
            return subprocess.CompletedProcess(cmd, 0, " x | 1 +\n", "")
        raise AssertionError(f"unexpected git call {cmd}")

    return run


@pytest.mark.parametrize("behaviour,expected", [
    ("", False),                    # clean tree
    (" M src/x.py\n", True),        # dirty tree
    ("timeout", None),              # git status timed out: unknown, not clean
    ("fails", None),                # git status failed: unknown, not clean
    ("oserror", None),              # git missing: unknown, not clean
])
def test_git_dirty_is_none_when_git_status_does_not_answer(monkeypatch, behaviour, expected):
    monkeypatch.setattr(provenance.subprocess, "run", _fake_git(behaviour))
    prov = provenance.collect_provenance()["provenance"]
    assert prov["git_dirty"] is expected
    assert prov["git_commit"] == SHA[:7]
    assert prov["git_diff_stat"] == ("x | 1 +" if expected else "")


def test_the_git_timeout_is_the_tooling_timeout_not_5_s(monkeypatch):
    seen = []

    def run(cmd, **kwargs):
        seen.append(kwargs.get("timeout"))
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(provenance.subprocess, "run", run)
    provenance.collect_provenance()
    assert seen and set(seen) == {provenance.GIT_TIMEOUT_S} and provenance.GIT_TIMEOUT_S >= 120


@pytest.mark.parametrize("behaviour", ["timeout", "fails"])
def test_the_queue_refuses_a_file_whose_tree_state_is_unknown(monkeypatch, behaviour):
    """End of the chain: provenance None -> validate_result reports the cell."""
    monkeypatch.setattr(provenance.subprocess, "run", _fake_git(behaviour))
    prov = provenance.collect_provenance()["provenance"]
    prov.update(git_commit_full=SHA, cr_arm="geowalk", run_overrides={})
    res = {"status": "completed", "n_evaluated": 30, "all_scores": [0.1] * 30, "budget": 30,
           "solver": "geowalk", "task": "ranolazine_mpo", "seed": 42, "subspace_dim": 16,
           "provenance": prov}
    cell = {"arm": "geowalk", "solver": "geowalk", "poli_task": "ranolazine_mpo", "seed": 42,
            "subspace_dim": 16, "run_overrides": {}}
    problems = cr_queue.validate_result(res, cell, expect_commit=SHA)
    assert problems == ["provenance git_dirty=None"]
    prov["git_dirty"] = False
    assert cr_queue.validate_result(res, cell, expect_commit=SHA) == []
