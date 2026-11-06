"""Cell masks and comparisons over a loaded Dataset, with file-set provenance."""
from __future__ import annotations

import hashlib
import os

import numpy as np

from . import stats
from .loader import Dataset

_SHA_CACHE: dict = {}


def file_sha256(path: str) -> str:
    """sha256 of a file's bytes, cached per (path, size, mtime)."""
    st = os.stat(path)
    key = (path, st.st_size, st.st_mtime_ns)
    if key not in _SHA_CACHE:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        _SHA_CACHE[key] = h.hexdigest()
    return _SHA_CACHE[key]


def fileset_from_rows(rows, arms=None) -> dict:
    """{count, list_sha256, content_sha256, arms, _rows} from (path, sha256) rows (deduplicated, sorted).
    The serialisation is the one GEN_verify_2 reproduced: list = paths joined by '\\n';
    content = 'path\\tsha' joined by '\\n'. '_rows' stays in memory (unions of derived numbers)."""
    rows = sorted(set(rows))
    return {
        "count": len(rows),
        "list_sha256": hashlib.sha256("\n".join(r[0] for r in rows).encode()).hexdigest(),
        "content_sha256": hashlib.sha256("\n".join(f"{p}\t{s}" for p, s in rows).encode()).hexdigest(),
        "arms": sorted(arms or []),
        "_rows": rows,
    }


def fileset_from_paths(paths, arms=None) -> dict:
    return fileset_from_rows([(p, file_sha256(p)) for p in paths], arms)


def fileset_union(*filesets) -> dict:
    """Union of file sets (a derived number inherits the files of everything it is derived from)."""
    rows, arms = [], set()
    for fs in filesets:
        if not fs:
            continue
        if "_rows" not in fs:
            raise ValueError("file set without rows cannot be merged")
        rows.extend(fs["_rows"])
        arms.update(fs.get("arms", []))
    return fileset_from_rows(rows, arms)


class Engine:
    def __init__(self, ds: Dataset, conv: str = "taskmean", cell_filter: str | None = None):
        self.ds = ds
        self.conv = conv
        self.cell_filter = cell_filter or ds.cell_filter
        self._elig: dict = {}

    # ---- arms and masks ---------------------------------------------------------------
    def has(self, arm: str) -> bool:
        return self.ds.present(arm)

    def elig(self, arm: str, cell_filter: str | None = None) -> set:
        f = cell_filter or self.cell_filter
        key = (arm, f)
        if key not in self._elig:
            self._elig[key] = self.ds.arm(arm).eligible(f)
        return self._elig[key]

    def joint(self, arms, cell_filter: str | None = None) -> set:
        arms = list(arms)
        out = set(self.elig(arms[0], cell_filter))
        for a in arms[1:]:
            out &= self.elig(a, cell_filter)
        return out

    def vals(self, arm: str, cells, N: int = 30) -> dict:
        a = self.ds.arm(arm)
        out = {k: a.value(k, N) for k in cells}
        bad = [k for k, v in out.items() if not np.isfinite(v)]
        if bad:
            raise ValueError(f"{arm}: non-finite value at N={N} in cells {sorted(bad)[:5]}")
        return out

    # ---- statistics ---------------------------------------------------------------------
    def level(self, arm: str, cells, N: int = 30, conv: str | None = None) -> float:
        return stats.level(self.vals(arm, cells, N), cells, conv or self.conv)

    def task_means(self, arm: str, cells, N: int = 30) -> dict:
        return stats.task_means(self.vals(arm, cells, N), cells)

    def paired(self, a: str, b: str, cells, N: int = 30, conv: str | None = None) -> stats.Paired:
        return stats.paired(self.vals(a, cells, N), self.vals(b, cells, N), cells, conv or self.conv)

    def paired_values(self, x: dict, y: dict, cells, conv: str | None = None) -> stats.Paired:
        return stats.paired(x, y, cells, conv or self.conv)

    # ---- provenance --------------------------------------------------------------------
    def fileset(self, arm_cells: dict) -> dict:
        """arm -> cells  ==>  {count, list_sha256, content_sha256, arms}. A cell recorded only by a
        queue crash sidecar (no result file) contributes the sidecar file."""
        rows = []
        for arm, cells in arm_cells.items():
            a = self.ds.arm(arm)
            for k in cells:
                c = a.cells.get(k)
                if c is not None:
                    rows.append((c.file, c.sha256))
                elif k in a.sidecar_crashes:
                    s = a.sidecar_crashes[k]
                    rows.append((s["path"], s["sha256"]))
                else:
                    raise KeyError(f"{arm}: cell {k} has neither a result file nor a crash sidecar")
            if a.spec.fmt == "external" and a.spec.coldstart_from:
                pass   # the cold-start prefix file is part of the coldstart arm's own set
        return fileset_from_rows(rows, arm_cells)

    def fileset_all(self, arm: str) -> dict:
        """Every file loaded for an arm (results and crash sidecars)."""
        a = self.ds.arm(arm)
        return self.fileset({arm: set(a.cells) | set(a.sidecar_crashes)})

    def generation(self, arm: str) -> dict:
        a = self.ds.arm(arm)
        return {"kind": a.spec.kind, "dir": a.spec.dir, "pattern": a.spec.pattern,
                "provenance_states": a.report.get("provenance_states")}
