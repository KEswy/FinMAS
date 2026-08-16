"""Walk-forward evaluation of the v6 graph orchestration."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from finmas.graph import FinMASGraph
from finmas.pipeline import FinMASPipeline
from finmas.schemas import EventInput


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--llm", action="store_true")
    parser.add_argument("--debate-rounds", type=int, default=2)
    parser.add_argument("--tag", default="v6_graph")
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args()

    car = pd.read_csv("data/processed/car_results_expanded.csv")
    car = car[car["window"] == 5].drop_duplicates(["event_date", "industry_code"])
    car["event_date"] = car["event_date"].astype(str).str[:10]
    car["industry_code"] = car["industry_code"].astype(str)
    car = car.sort_values("event_date").reset_index(drop=True)
    if args.start:
        car = car.iloc[int(args.start) :]
    if args.limit:
        car = car.head(int(args.limit))

    pipeline = FinMASPipeline(use_llm=args.llm, horizon=5)
    graph = FinMASGraph(pipeline=pipeline, debate_rounds=args.debate_rounds)
    rows = []

    for i, (_, row) in enumerate(car.iterrows()):
        event = EventInput(
            event_date=row["event_date"],
            event_text=str(row["event_name"]),
            event_type=str(row["event_type"]),
            industry_code=row["industry_code"],
        )
        try:
            state = graph.run(event)
            decision = state.decision
            final_dir = decision.final_direction if decision else "0"
            real_dir = "+" if float(row["CAR"]) >= 0 else "-"
            rows.append(
                {
                    "event_date": row["event_date"],
                    "event_type": row["event_type"],
                    "industry_code": row["industry_code"],
                    "real_dir": real_dir,
                    "final_dir": final_dir,
                    "prob_up": decision.prob_up if decision else float("nan"),
                    "abstain": decision.abstain if decision else True,
                    "dir_correct": bool(final_dir == real_dir),
                    "committed_dir_correct": bool(
                        decision and not decision.abstain and final_dir == real_dir
                    ),
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "event_date": row["event_date"],
                    "event_type": row["event_type"],
                    "industry_code": row["industry_code"],
                    "real_dir": "+" if float(row["CAR"]) >= 0 else "-",
                    "final_dir": "0",
                    "prob_up": float("nan"),
                    "abstain": True,
                    "dir_correct": False,
                    "committed_dir_correct": False,
                }
            )
            print(f"error {row['event_date']}/{row['industry_code']}: {exc}")

        if (i + 1) % max(args.log_every, 1) == 0:
            df_so_far = pd.DataFrame(rows)
            committed = df_so_far[~df_so_far["abstain"]]
            print(
                f"[graph eval] processed={i + 1}/{len(car)} "
                f"full_acc={df_so_far['dir_correct'].mean():.4f} "
                f"committed_acc={committed['committed_dir_correct'].mean():.4f} "
                f"coverage={len(committed)/max(len(df_so_far),1):.4f}",
                flush=True,
            )

    df = pd.DataFrame(rows)
    out = Path("data/processed") / f"all_experiment_results_{args.tag}_walk_forward.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    committed = df[~df["abstain"]]
    print(
        f"rows={len(df)} full_acc={df['dir_correct'].mean():.4f} "
        f"committed_acc={committed['committed_dir_correct'].mean():.4f} "
        f"coverage={len(committed)/max(len(df),1):.4f} output={out}"
    )


if __name__ == "__main__":
    main()
