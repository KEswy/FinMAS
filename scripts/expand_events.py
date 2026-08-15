r"""
Merge the 46 web-verified NEW policy events into an EXPANDED event library.
- Original data/events/event_library.csv is left UNTOUCHED.
- New events carry EMPTY direction/magnitude (option A): under the firewall the
  label is the CAR sign computed from prices; no human direction ever enters.
- Industries come from the FROZEN keyword map (assign_industries.py) -> 5c-clean.
- Leak-free names were already checked (0 banned words) -> 5b-clean.

Writes data/events/event_library_expanded.csv  (50 original + 46 new = 96).
Does NOT compute CAR (run compute_car.py after, pointed at the expanded file).
"""
from __future__ import annotations
import os, sys, io, csv
if sys.platform == "win32" and hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(ROOT)

ORIG = "data/events/event_library.csv"
CAND = "data/events/_candidate_events_mapped.csv"
OUT = "data/events/event_library_expanded.csv"
COLS = ["event_date", "event_type", "event_name", "affected_industries",
        "direction", "magnitude", "description"]


def main():
    orig = list(csv.DictReader(open(ORIG, encoding="utf-8-sig")))
    cand = list(csv.DictReader(open(CAND, encoding="utf-8-sig")))
    orig_dates = {r["event_date"] for r in orig}

    merged = [dict(r) for r in orig]        # keep originals verbatim
    added = 0
    for r in cand:
        if r["event_date"] in orig_dates:
            continue                        # never overwrite an original event
        if not r["affected_industries"]:
            continue                        # frozen map found no industry -> drop
        merged.append({
            "event_date": r["event_date"],
            "event_type": r["event_type"],
            "event_name": r["event_name"],
            "affected_industries": r["affected_industries"],
            "direction": "",                # option A: empty, never hand-set
            "magnitude": "",
            "description": r.get("notes", ""),
        })
        added += 1

    merged.sort(key=lambda r: r["event_date"])
    with open(OUT, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
        w.writeheader()
        for r in merged:
            w.writerow({c: r.get(c, "") for c in COLS})

    print(f"original events: {len(orig)}")
    print(f"new events added: {added}")
    print(f"total in {OUT}: {len(merged)}")
    # event_type breakdown
    from collections import Counter
    c = Counter(r["event_type"] for r in merged)
    print("event_type counts:", dict(c))


if __name__ == "__main__":
    main()
