"""
Phase 0 walk-forward smoke test — ONE event, confirms the firewall runs
through the real pipeline before committing to a multi-hour full run.

Watch the log for TWO firewall lines:
  [walk-forward] 检索器已按 event_date<2024-07-22 过滤，可见文档=...
  [walk-forward] 敏感度先验基于 N 个历史事件（<2024-07-22），... 个非平凡 cell

Run (PowerShell, in v4 dir):
    $env:WALK_FORWARD="1"; $env:SEED="42"; python smoke_walk_forward.py
"""
import os
os.environ.setdefault("WALK_FORWARD", "1")

import main

r = main.run_analysis(
    event="LPR双降：1年期降至3.35%(-10bp)，5年期降至3.85%(-10bp)",
    entity_names=["LPR双降", "801780"],
    task_type="policy_event",
    industry_code="801780",
    event_date="2024-07-22",
    walk_forward=True,
)

pf = r["prediction"]["point_forecast_real"][:5]
pred_dir = "+" if sum(pf) / len(pf) > 0 else "-"
print("\n" + "=" * 50)
print("WF SMOKE OK")
print("  pred_dir  =", pred_dir)
print("  pred[:5]  =", [round(x, 5) for x in pf])
print("  triggered =", r.get("triggered_correction"))
print("=" * 50)
