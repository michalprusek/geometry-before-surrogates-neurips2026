"""Print run metadata of verification records WITHOUT any objective value (blinding: counts and settings only).
usage: check_records.py <out dir> [expected commit]"""
import json, sys, pathlib
out = pathlib.Path(sys.argv[1]); exp = sys.argv[2] if len(sys.argv) > 2 else None
KEYS = ("recipe", "benchmark", "seed", "status", "n_init", "init_type", "budget", "batch_size", "m_d", "fallback_count")
bad = 0
for p in sorted(out.glob("*.json")):
    try:
        r = json.loads(p.read_text())
    except ValueError:
        print(p.name, "UNREADABLE"); bad += 1; continue
    if not isinstance(r, dict) or "recipe" not in r:
        continue
    prov = r.get("provenance") or {}
    row = {k: r.get(k) for k in KEYS}
    row["n_values"] = len(r.get("values") or [])
    row["commit"] = (prov.get("git_commit_full") or "")[:7]; row["dirty"] = prov.get("git_dirty")
    ok = (exp is None or prov.get("git_commit_full") == exp) and prov.get("git_dirty") is False
    bad += not ok
    print(("OK  " if ok else "BAD ") + " ".join(f"{k}={v}" for k, v in row.items()))
print("records with provenance problems:", bad)
