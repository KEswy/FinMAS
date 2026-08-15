"""
Split the v3 doc_library into a firewall-clean event corpus + a quarantined
mechanism set (v4 Phase 0, leak channel L4).

Problem
-------
The 92-doc corpus mixes two kinds of documents:
  - 74 EVENT-TIED docs (doc_type in {policy, research, news, analysis}), each
    with a real event_date/pub_date. These are firewalled by pub_date < t.
  - 18 MECHANISM docs (doc_type == "mechanism", event_date == ""), which look
    "timeless" but in fact quote FORWARD-LOOKING realized statistics spanning
    2020-2024 (e.g. "统计 2020-2024 年 LPR 调降日 ... 60% 样本已被定价 ... 公告日
    获利了结"). Any pre-2024 event retrieving these reads future outcomes — a
    genuine leak, more insidious than the direction label.

Decision (user, 2026-07-14): QUARANTINE the 18 mechanism docs — the honest,
zero-leak, fastest option. They are written out separately, tagged
``undated_origin: true`` so the firewall drops them by default (see
firewall.FirewalledRetriever._visible). They are NOT deleted: kept for a
possible Phase-3 rewrite into genuinely time-invariant mechanism knowledge.

Outputs (in data/events/):
  doc_library_events.json      -> firewall-clean event corpus (default RAG source)
  mechanism_quarantine.json    -> the 18 quarantined docs, tagged undated_origin

Run:  python scripts/split_doc_library.py
"""
from __future__ import annotations

import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
os.chdir(ROOT_DIR)

SRC = "data/events/doc_library.json"
OUT_EVENTS = "data/events/doc_library_events.json"
OUT_QUARANTINE = "data/events/mechanism_quarantine.json"


def main() -> int:
    with open(SRC, encoding="utf-8") as f:
        recs = json.load(f)

    events, mechanism = [], []
    for r in recs:
        if r.get("doc_type") == "mechanism" or not r.get("event_date"):
            # tag so the firewall drops it by default, but preserve the record
            r = dict(r)
            r["undated_origin"] = True
            mechanism.append(r)
        else:
            events.append(r)

    with open(OUT_EVENTS, "w", encoding="utf-8") as f:
        json.dump(events, f, ensure_ascii=False, indent=2)
    with open(OUT_QUARANTINE, "w", encoding="utf-8") as f:
        json.dump(mechanism, f, ensure_ascii=False, indent=2)

    print(f"[split] source total     : {len(recs)}")
    print(f"[split] event-tied  -> {OUT_EVENTS} : {len(events)}")
    print(f"[split] quarantined -> {OUT_QUARANTINE} : {len(mechanism)}")
    # sanity: report pub_date spans
    def span(rs):
        ds = sorted(d for d in (x.get("pub_date", "") for x in rs) if d)
        return (ds[0], ds[-1]) if ds else ("-", "-")
    print(f"[split] event pub_date span     : {span(events)}")
    print(f"[split] mechanism pub_date span : {span(mechanism)}")
    if len(events) + len(mechanism) != len(recs):
        print("[split] WARN: counts do not add up", file=sys.stderr)
        return 1
    print("[split] OK — default RAG source is now doc_library_events.json "
          "(set DOC_LIBRARY_PATH to override).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
