"""retrieval_floor.py: write_md renders a JSON with the renamed provenance keys (floor_worktree_*) and one
with the old keys (e0_*); the write guard refuses input directories and git work trees by path component."""
import importlib.util
import os
import shutil
import subprocess

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(os.path.dirname(HERE), "retrieval_floor.py")


@pytest.fixture(scope="module")
def rf():
    spec = importlib.util.spec_from_file_location("retrieval_floor_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def minimal_result(rf, keys: str) -> dict:
    pv = {"finished": "2026-10-12T13:24:26", "host": "testhost", "script": "/x/retrieval_floor.py",
          "script_sha256": "a" * 64, "python": "3.12", "numpy": "2.4", "scipy": "1.17",
          "floor_module": "/x/zinc_retrieval_floor.py", "floor_module_sha256": "b" * 64,
          "smiles_sha256_now": "c" * 64, "repo_head": "14481cb55d21a770", "repo_status_lines": 0,
          "repo_status_lines_after": 0}
    if keys == "new":
        pv.update(floor_worktree_head="4780475754bbda11", floor_worktree_status_lines=3,
                  floor_worktree_status_lines_after=3)
    else:
        pv.update(e0_worktree_head="4780475754bbda11", e0_worktree_status_lines=3, e0_status_lines_after=3)
    floors = {t: {"floor_norepl": 0.4, "floor_replace": 0.41, "sd_max30_norepl": 0.05,
                  "max30_quantiles_norepl": {"q05": 0.3, "q50": 0.4, "q95": 0.5}, "null40_q025": 0.38,
                  "null40_q975": 0.42, "library_boot_se": 0.001, "floor_10k_prefix_norepl": 0.39,
                  "floor_3k_latent_pool_norepl": 0.37, "scored_commit": "c33f51d", "scored_dirty": False}
              for t in rf.TASKS}
    return {"provenance": pv, "floors": floors, "mean_floor_norepl": 0.4, "mean_floor_replace": 0.41, "arms": {}}


@pytest.mark.parametrize("keys", ["new", "old"])
def test_write_md_renders_new_and_old_provenance_keys(rf, tmp_path, keys):
    out = tmp_path / f"{keys}.md"
    rf.write_md(minimal_result(rf, keys), str(out))           # raised KeyError: 'e0_worktree_head' for 'new'
    text = out.read_text()
    assert "e0 worktree HEAD `4780475`, 3 status lines" in text
    assert "Repo `14481cb` status lines before/after: 0/0" in text
    assert text.count("\n| ") >= len(rf.TASKS)                # one floor row per task


def test_both_key_spellings_render_the_same_text(rf, tmp_path):
    rf.write_md(minimal_result(rf, "new"), str(tmp_path / "a.md"))
    rf.write_md(minimal_result(rf, "old"), str(tmp_path / "b.md"))
    assert (tmp_path / "a.md").read_text() == (tmp_path / "b.md").read_text()


def test_missing_key_in_both_spellings_is_an_error(rf, tmp_path):
    R = minimal_result(rf, "new")
    del R["provenance"]["floor_worktree_head"]
    with pytest.raises(KeyError) as e:
        rf.write_md(R, str(tmp_path / "c.md"))
    assert "floor_worktree_head" in str(e.value) and "e0_worktree_head" in str(e.value)


def test_aliases_are_written_into_the_json_provenance(rf):
    pv = {"floor_worktree_head": "abc", "floor_worktree_status_lines": 0, "floor_worktree_status_lines_after": 1}
    rf._add_aliases(pv)
    assert pv["e0_worktree_head"] == "abc" and pv["e0_worktree_status_lines"] == 0 and pv["e0_status_lines_after"] == 1
    assert pv["provenance_key_aliases"] == rf.PROVENANCE_ALIASES


def test_write_guard_compares_path_components_and_refuses_git_work_trees(rf, tmp_path):
    a = tmp_path / "NeurIPS"
    b = tmp_path / "NeurIPS-cr"
    a.mkdir()
    b.mkdir()
    assert rf._inside(str(a / "results" / "x"), str(a)) and rf._inside(str(a), str(a))
    assert not rf._inside(str(b / "out"), str(a))             # '/x/NeurIPS-cr' is not inside '/x/NeurIPS'
    with pytest.raises(SystemExit) as e:
        rf.check_outdir(str(a / "out"), [str(a), "", str(b)])
    assert "input directory" in str(e.value)
    if not shutil.which("git") or rf._git_toplevel(str(tmp_path)):
        pytest.skip("git missing, or pytest's temporary directory is inside a git work tree")
    rf.check_outdir(str(b / "out"), [str(a)])                 # a sibling with a common prefix is allowed
    repo = tmp_path / "clone"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    with pytest.raises(SystemExit) as e:
        rf.check_outdir(str(repo / "results" / "floor" / "new"), [])
    assert "git work tree" in str(e.value)
