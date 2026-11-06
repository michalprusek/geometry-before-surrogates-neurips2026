"""tools/sync_lassobench_cr.sh: the LassoBench rerun counts as finished only when BOTH done flags exist
(queue/full_rerun.done: the original 1,442 jobs; queue/papenmeier_leukemia.done: the 39 Papenmeier leukemia jobs of
the RUN_RULES amendment of 2026-10-12); the tool names the missing flag, and a done flag reaches the destination
only next to results that were copied after its jobs had ended.

The two servers are local folders here: an ``ssh`` stand-in on PATH drops the options and the host and runs the
remote command on this machine, so the real script and the real sync_set.sh are exercised end to end."""
import os
import shutil
import stat
import subprocess

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
TOOL = os.path.join(os.path.dirname(HERE), "tools", "sync_lassobench_cr.sh")
FLAGS = ("full_rerun.done", "papenmeier_leukemia.done")
SSH_STANDIN = """#!/bin/bash
[ -n "${STANDIN_FAIL:-}" ] && exit 255                        # a dropped connection
while [ $# -gt 0 ]; do case "$1" in -o) shift 2;; -*) shift;; *) break;; esac; done
shift                                                          # the host
case "$*" in *"tar -cf"*) [ -n "${STANDIN_TOUCH_ON_TAR:-}" ] && touch "$STANDIN_TOUCH_ON_TAR";; esac
exec bash -c "$*"
"""

pytestmark = pytest.mark.skipif(not all(shutil.which(x) for x in ("bash", "sha256sum", "tar", "find", "xargs")),
                                reason="needs bash, sha256sum, tar, find and xargs")


class Servers:
    def __init__(self, tmp_path):
        self.src = tmp_path / "kajman" / "lassobench_cr"
        self.queue = tmp_path / "kajman" / "cr_analysis"
        self.dst = tmp_path / "panda" / "cr_inputs"
        files = {self.src / "full_rerun" / "results" / "papenmeier_msr_leukemia_s42.json": "{}",
                 self.src / "full_rerun" / "manifest.jsonl": "{}\n",
                 self.src / "full_rerun" / "logs" / "papenmeier_leukemia_s43.log": "still being written",
                 self.src / "leukemia_rerun" / "results" / "geowalk_leukemia_s42.json": "{}",
                 self.src / "linear_bo_leukemia" / "PILOT_FAILED": "",
                 self.src / "env_provenance.txt": "env",
                 self.src / "bitcheck_d60" / "compare.txt": "cmp",
                 self.queue / "queue" / "jobs_full_rerun.txt": "1442 lines\n",
                 self.queue / "queue" / "full_rerun.log": "log\n",
                 self.queue / "queue" / "jobs_papenmeier_leukemia.txt": "39 lines\n",
                 self.queue / "queue" / "main.done": "another queue: not copied\n",
                 self.queue / "code" / "run_lasso.sh": "#!/bin/bash\n"}
        for p, text in files.items():
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)
        self.bin = tmp_path / "bin"
        self.bin.mkdir()
        ssh = self.bin / "ssh"
        ssh.write_text(SSH_STANDIN)
        ssh.chmod(ssh.stat().st_mode | stat.S_IXUSR)
        self.tmp = tmp_path / "tmp"
        self.tmp.mkdir()

    def flag(self, name):
        (self.queue / "queue" / name).write_text("done\n")

    def run(self, *args, **env):
        e = dict(os.environ, PATH=f"{self.bin}{os.pathsep}{os.environ['PATH']}", TMPDIR=str(self.tmp),
                 LASSO_SRC_DIR=str(self.src), LASSO_QUEUE_DIR=str(self.queue), LASSO_DST_PARENT=str(self.dst),
                 LASSO_SYNC_TRIES="1", LASSO_SYNC_WAIT="0", **env)
        r = subprocess.run(["bash", TOOL, *args], capture_output=True, text=True, env=e, timeout=300)
        return r.returncode, r.stdout, r.stderr

    def copied_flags(self):
        q = self.dst / "lassobench_cr_queue" / "queue"
        return tuple(f for f in FLAGS if (q / f).exists())

    def writable(self):                       # sync_set.sh leaves the copy read-only; let pytest remove it
        subprocess.run(["chmod", "-R", "u+w", str(self.dst)], check=False)


@pytest.mark.parametrize("present", [(), ("full_rerun.done",), ("papenmeier_leukemia.done",), FLAGS])
def test_finished_only_with_both_done_flags_and_the_missing_one_is_named(tmp_path, present):
    S = Servers(tmp_path)
    for f in present:
        S.flag(f)
    missing = [f for f in FLAGS if f not in present]
    # --check: reports the flags, copies nothing
    rc, out, err = S.run("--check")
    assert rc == (3 if missing else 0) and not S.dst.exists()
    if missing:
        assert f"LassoBench rerun NOT FINISHED: missing done flag(s): {' '.join(missing)}" in out
    else:
        assert "LassoBench rerun FINISHED: both done flags exist" in out
    for f in FLAGS:
        assert sum(1 for ln in out.splitlines() if f"/queue/{f} " in ln and ("MISSING" if f in missing else "present") in ln) == 1
    if "papenmeier_leukemia.done" in missing:
        assert "(the 39 Papenmeier leukemia jobs, seeds 43-81 have not ended)" in out
    if "full_rerun.done" in missing:
        assert "(the original 1,442 jobs of the full rerun have not ended)" in out
    # the copy: verified in every state, FINAL (exit 0) only with both flags
    rc, out, err = S.run()
    try:
        assert rc == (3 if missing else 0), err
        assert (S.dst / "lassobench_cr" / "full_rerun" / "results" / "papenmeier_msr_leukemia_s42.json").exists()
        assert (S.dst / "lassobench_cr" / "linear_bo_leukemia" / "PILOT_FAILED").exists()
        assert not (S.dst / "lassobench_cr" / "full_rerun" / "logs").exists()             # logs stay on the source
        q = S.dst / "lassobench_cr_queue" / "queue"
        assert (q / "jobs_papenmeier_leukemia.txt").read_text() == "39 lines\n" and (q / "jobs_full_rerun.txt").exists()
        assert not (q / "main.done").exists()
        assert S.copied_flags() == tuple(present)
        man = (S.dst / "lassobench_cr_queue.sha256").read_text()
        assert all((f"./queue/{f}" in man) == (f in present) for f in FLAGS)
        assert man == (S.dst / "SOURCE_lassobench_cr_queue.sha256").read_text()
        if missing:
            assert "done, PARTIAL copy" in err and "NOT finished" in err and "FINAL copy" not in out
            assert all(f"/queue/{f} " in err for f in missing) and not any(f"/queue/{f} " in err for f in present)
        else:
            assert "done, FINAL copy (both done flags existed before the copy started)" in out and "PARTIAL" not in err
    finally:
        S.writable()


def test_a_done_flag_written_during_the_copy_is_not_copied(tmp_path):
    """The last jobs end while the results are being copied: the copy is partial and must not carry the flag."""
    S = Servers(tmp_path)
    S.flag("full_rerun.done")
    late = S.queue / "queue" / "papenmeier_leukemia.done"
    rc, out, err = S.run(STANDIN_TOUCH_ON_TAR=str(late))
    try:
        assert late.exists() and rc == 3, err                    # on the source the flag now exists ...
        assert S.copied_flags() == ("full_rerun.done",)          # ... but it did not when the copy started
        assert "missing done flag(s): papenmeier_leukemia.done" in out and "PARTIAL copy" in err
        rc, out, err = S.run()                                   # the next run is the final copy
        assert rc == 0 and S.copied_flags() == FLAGS and "FINAL copy" in out, err
    finally:
        S.writable()


def test_unreadable_flags_and_bad_usage_copy_nothing(tmp_path):
    S = Servers(tmp_path)
    for f in FLAGS:
        S.flag(f)
    rc, out, err = S.run(STANDIN_FAIL="1")                       # no answer from the source host is not "missing"
    assert rc == 1 and "could not be read; nothing was copied" in err and "FINISHED" not in out and not S.dst.exists()
    rc, out, err = S.run("--chekc")
    assert rc == 2 and "usage" in err and not S.dst.exists()
