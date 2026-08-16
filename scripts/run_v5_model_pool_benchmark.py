"""Walk-forward benchmark for the v5 time-series model pool."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from finmas.data.feature_store import FeatureStore
from finmas.models.pool import ModelPool


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--tag", default="v5_model_pool")
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args()

    store = FeatureStore()
    car = pd.read_csv("data/processed/car_results_expanded.csv")
    car = car[car["window"] == 5].drop_duplicates(["event_date", "industry_code"])
    car["event_date"] = car["event_date"].astype(str).str[:10]
    car["industry_code"] = car["industry_code"].astype(str)
    car = car.sort_values("event_date").reset_index(drop=True)
    if args.start:
        car = car.iloc[int(args.start) :]
    if args.limit:
        car = car.head(int(args.limit))

    rows = []
    for i, (_, row) in enumerate(car.iterrows()):
        matrix = store.pre_event_matrix(
            row["industry_code"], row["event_date"], lookback=600
        )
        if matrix is None or len(matrix) < 90:
            continue
        x = matrix.to_numpy(dtype=float)
        y = x[:, 0:1]
        pool = ModelPool(horizon=5, input_len=60, output_len=5)
        try:
            pool.fit(x, y)
            pred = pool.predict(x[-60:])
            pred_mean5 = float(np.mean(pred.mean))
        except Exception:
            continue
        real_dir = "+" if float(row["CAR"]) >= 0 else "-"
        final_dir = "+" if pred_mean5 >= 0 else "-"
        rows.append(
            {
                "event_date": row["event_date"],
                "event_type": row["event_type"],
                "industry_code": row["industry_code"],
                "real_dir": real_dir,
                "final_dir": final_dir,
                "dir_correct": bool(final_dir == real_dir),
                "pred_mean_5": pred_mean5,
                "real_CAR": float(row["CAR"]),
            }
        )
        if (i + 1) % max(args.log_every, 1) == 0:
            acc = float(np.mean([r["dir_correct"] for r in rows]))
            print(
                f"[model pool] processed={i + 1}/{len(car)} acc={acc:.4f}",
                flush=True,
            )

    df = pd.DataFrame(rows)
    out = Path("data/processed") / f"all_experiment_results_{args.tag}_walk_forward.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"rows={len(df)} full_acc={df['dir_correct'].mean():.4f} output={out}")


if __name__ == "__main__":
    main()

