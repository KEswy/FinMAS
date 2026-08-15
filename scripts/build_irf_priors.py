"""
Assemble data/irf_priors.json from the adversarially-verified transmission
directions produced by the irf-prior-build workflow (Phase 3 prep).

Reads the workflow journal (5 verify verdicts, one per event_type), matches
each verdict set to its event_type by industry-code fingerprint, attaches the
industry name, and writes a clean, documented IRF prior table.

The prior carries: direction (sign to the canonical shock), strength (signed,
[-1,1], shrunk under theory/data conflict or thin n), and the verifier's reason.
Direction/strength come from transmission ECONOMICS (verified); the historical
CAR was used only to calibrate strength. At prediction time the transmission
skill will still respect the firewall (strength recalibrated on <t data).

Run:  python scripts/build_irf_priors.py
"""
from __future__ import annotations
import os, sys, json, glob
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- event_type -> its exact industry-code set (from irf_skeleton) ----
# fingerprint each event_type by the sorted set of industry codes it covers
SKELETON = "data/processed/irf_skeleton.csv"
import csv
et_codes: dict[str, set] = {}
et_names: dict[str, dict] = {}
with open(SKELETON, encoding="utf-8-sig") as f:
    for row in csv.DictReader(f):
        et = row["event_type"]; ic = row["industry_code"]
        et_codes.setdefault(et, set()).add(ic)
        et_names.setdefault(et, {})[ic] = row["industry_name"]

# --- locate the workflow journal ------------------------------------
cand = glob.glob(os.path.join(
    os.path.expanduser("~"),
    ".claude/projects/*/**/workflows/wf_*/journal.jsonl"), recursive=True)
# prefer the irf-prior-build run if multiple
journal = None
for p in sorted(cand, key=os.path.getmtime, reverse=True):
    txt = open(p, encoding="utf-8").read()
    if '"verdicts"' in txt:
        journal = p; break
if not journal:
    print("ERROR: no workflow journal with verdicts found", file=sys.stderr)
    sys.exit(1)
print(f"[irf] journal: {journal}")

# --- collect all verify verdict sets --------------------------------
verify_sets = []
for line in open(journal, encoding="utf-8"):
    try:
        o = json.loads(line)
    except Exception:
        continue
    if o.get("type") != "result":
        continue
    r = o.get("result", {})
    if "verdicts" in r and r["verdicts"]:
        verify_sets.append(r["verdicts"])

print(f"[irf] found {len(verify_sets)} verify sets")

# --- match each verdict set to its event_type by code fingerprint ---
priors = {}
used_ets = set()
for vs in verify_sets:
    codes = {v["industry_code"] for v in vs}
    # match event_type by EXACT code-set equality (each event_type has a
    # unique industry-code fingerprint; greedy max-overlap misassigns because
    # codes like 801780 are shared across event types).
    best_et = None
    for et, ecodes in et_codes.items():
        if et in used_ets:
            continue
        if codes == ecodes:
            best_et = et
            break
    if best_et is None:
        # fallback: strict subset/superset then max Jaccard, still unused
        best_j = -1.0
        for et, ecodes in et_codes.items():
            if et in used_ets:
                continue
            j = len(codes & ecodes) / max(len(codes | ecodes), 1)
            if j > best_j:
                best_j, best_et = j, et
    if best_et is None:
        continue
    used_ets.add(best_et)
    cells = {}
    for v in vs:
        ic = v["industry_code"]
        cells[ic] = {
            "industry_name": et_names.get(best_et, {}).get(ic, "?"),
            "direction": int(v["final_direction"]),
            "strength": round(float(v["final_strength"]), 3),
            "reason": v.get("reason", ""),
            "verified_agree": bool(v.get("agree", True)),
        }
    priors[best_et] = cells

# --- write ----------------------------------------------------------
out = {
    "_meta": {
        "source": "irf-prior-build workflow (propose + adversarial verify)",
        "semantics": "direction/strength = industry response sign & signed prior "
                     "to the canonical shock of each event_type, from transmission "
                     "economics; strength shrunk under theory/data conflict or thin n. "
                     "Historical CAR used only to calibrate strength. Prediction-time "
                     "use must still respect the temporal firewall.",
        "n_event_types": len(priors),
        "n_cells": sum(len(c) for c in priors.values()),
    },
    "priors": priors,
}
outpath = "data/irf_priors.json"
with open(outpath, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)

print(f"[irf] wrote {outpath}: {len(priors)} event_types, "
      f"{out['_meta']['n_cells']} cells")
for et, cells in priors.items():
    npos = sum(1 for c in cells.values() if c["direction"] > 0)
    nneg = sum(1 for c in cells.values() if c["direction"] < 0)
    nzero = sum(1 for c in cells.values() if c["direction"] == 0)
    print(f"  {et:<24} {len(cells):>2} cells  (+{npos} / -{nneg} / 0:{nzero})")
