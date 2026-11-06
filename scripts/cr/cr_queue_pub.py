#!/usr/bin/env python
"""Queue driver for the published-recipe block (wave ``p``, ``scripts/cr/manifest_pub.json``).

This is ``scripts/cr/cr_queue.py`` unchanged, with two differences that exist only inside this
process:

* wave ``p`` is admitted (``cr_queue.VALID_WAVES`` is extended for the duration of the call and
  restored afterwards, so ``cr_queue.py`` and ``scripts/cr/manifest.json`` stay byte-identical for
  the running campaigns and their tests);
* the manifest defaults to ``scripts/cr/manifest_pub.json``, whose ``output_root`` is
  ``results/guacamol/cr2026pub``; any other EFFECTIVE output root (the one ``cr_queue``'s own
  parser takes, i.e. the last ``--output-root``, resolved against ``--repo``) is refused, except a
  smoke root ``results/guacamol/cr2026pub_smoke*`` (parent exactly ``results/guacamol``); a
  manifest with arms outside wave ``p`` is refused.

Every flag of ``cr_queue.py`` works as documented there, e.g.::

    python scripts/cr/cr_queue_pub.py --expect-commit <sha> --gpus 1 --workers-per-gpu 1 \\
        --tasks rano --seeds 42,43 --dry-run

The children are ``scripts/cr/run_cell.py`` with ``--manifest scripts/cr/manifest_pub.json``, so
``run_cell`` checks every cell against this manifest (arm, solver, overrides, seed range).
"""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.cr import cr_queue  # noqa: E402

PUB_MANIFEST = REPO_ROOT / "scripts" / "cr" / "manifest_pub.json"
PUB_WAVES = ("p",)
PUB_OUTPUT_ROOT = "results/guacamol/cr2026pub"
PUB_SMOKE_PREFIX = "cr2026pub_smoke"
PUB_SOLVERS = (
    "hvarfner_published",
    "papenmeier_msr_reference",
    "papenmeier_mle_scaled_reference",
    "xu_published",
    "linear_bo_box8",
    "turbo_published",
    "hvarfner_published_allobs",
    "papenmeier_msr_reference_learned",
    "papenmeier_mle_scaled_reference_learned",
)


@contextlib.contextmanager
def published_waves():
    """Admit wave ``p`` in ``cr_queue`` for the duration of the block; always restored."""
    old = cr_queue.VALID_WAVES
    cr_queue.VALID_WAVES = tuple(old) + tuple(w for w in PUB_WAVES if w not in old)
    try:
        yield
    finally:
        cr_queue.VALID_WAVES = old


def pub_manifest_problems(man: dict) -> list[str]:
    """Block-specific rules on top of ``cr_queue.validate_manifest``."""
    problems = []
    if man.get("output_root") != PUB_OUTPUT_ROOT:
        problems.append(f"output_root must be {PUB_OUTPUT_ROOT!r}, got {man.get('output_root')!r}")
    for arm in man.get("arms", []):
        if str(arm.get("wave")) not in PUB_WAVES:
            problems.append(f"arm {arm.get('arm')!r} is not in wave 'p'")
        if arm.get("solver") not in PUB_SOLVERS:
            problems.append(f"arm {arm.get('arm')!r} runs {arm.get('solver')!r}, not a "
                            f"published-recipe solver")
    return problems


def load_pub_manifest(path: Path = PUB_MANIFEST) -> dict:
    with published_waves():
        man = cr_queue.load_manifest(Path(path))
    problems = pub_manifest_problems(man)
    if problems:
        raise SystemExit(f"manifest {path}: " + "; ".join(problems))
    return man


def _flag_value(argv: list[str], flag: str):
    for i, a in enumerate(argv):
        if a == flag and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith(flag + "="):
            return a.split("=", 1)[1]
    return None


def effective_args(argv: list[str]):
    """``cr_queue``'s own parse of ``argv`` (last flag wins), with wave ``p`` admitted."""
    with published_waves():
        return cr_queue.build_parser().parse_args(argv)


def check_output_root(args) -> Path:
    """The effective output root must be ``<repo>/results/guacamol/cr2026pub`` or a smoke root
    ``<repo>/results/guacamol/cr2026pub_smoke*`` (round-1 fix C10). Returns it, else exits."""
    repo = Path(args.repo).resolve()
    man_root = PUB_OUTPUT_ROOT if args.output_root is None else str(args.output_root)
    root = Path(man_root)
    root = (root if root.is_absolute() else repo / root).resolve()
    main_root = (repo / PUB_OUTPUT_ROOT).resolve()
    smoke_parent = (repo / "results" / "guacamol").resolve()
    if root == main_root:
        return root
    if root.parent == smoke_parent and root.name.startswith(PUB_SMOKE_PREFIX):
        return root
    raise SystemExit(f"effective --output-root {root} is neither {main_root} nor a "
                     f"{smoke_parent}/{PUB_SMOKE_PREFIX}* smoke root (published block)")


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if _flag_value(argv, "--manifest") is None:
        argv = ["--manifest", str(PUB_MANIFEST), *argv]
    args = effective_args(argv)
    check_output_root(args)
    load_pub_manifest(Path(args.manifest))
    with published_waves():
        return cr_queue.main(argv)


if __name__ == "__main__":
    sys.exit(main())
