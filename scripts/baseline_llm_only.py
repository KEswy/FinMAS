"""
LLM-Only Baseline (Plan C, Stage 3)
====================================

Pure LLM end-to-end baseline. Given (event_description, industry_code, event_date),
ask the LLM directly to predict CAR sign (positive/negative) with no mechanism
chain, no RAG, no time-series predictor, no exogenous correction.

This isolates the LLM's pure event-understanding capacity, and provides an upper
bound on what end-to-end LLM-based finance forecasting could achieve without
the architectural pipeline we develop in the main paper.

The LLM-only prompt:
  "Below is a policy/macro event affecting Chinese A-share market.
   Event date: {date}
   Affected industry: {industry_name} ({industry_code})
   Event description: {event_text}

   Question: Will the cumulative abnormal return (CAR) of this industry
   over the 5 trading days following the event be positive or negative?
   Answer with exactly one character: + or -."

CLI:
  python scripts/baseline_llm_only.py
  $env:LLM_MODEL="qwen2.5:32b"; python scripts/baseline_llm_only.py   # explicit LLM
  $env:LLM_MODEL="llama3.1:8b"; python scripts/baseline_llm_only.py   # cross-LLM
"""
from __future__ import annotations
import os, sys, io, datetime, json, urllib.request, urllib.error, argparse

if sys.platform == "win32" and hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, ROOT_DIR)
os.chdir(ROOT_DIR)

import pandas as pd
import numpy as np
from agents.llm_client import chat_completion, get_llm_config


# Shenwan industry code → name mapping (top sectors used in our 88-event benchmark)
INDUSTRY_NAMES = {
    "801010": "农林牧渔",       "801030": "基础化工",      "801050": "有色金属",
    "801080": "电子",            "801110": "家用电器",      "801160": "公用事业",
    "801180": "房地产",          "801710": "建筑材料",      "801730": "电力设备",
    "801740": "国防军工",        "801750": "计算机",        "801760": "传媒",
    "801770": "通信",            "801780": "银行",          "801790": "非银金融",
    "801880": "汽车",            "801890": "机械设备",
}


def _call_llm(model: str, prompt: str, timeout: int = 90) -> str:
    """Single provider call for the LLM-only baseline."""
    return chat_completion(
        system="You are a financial analyst. Reply concisely.",
        user=prompt,
        model=model or get_llm_config()["model"],
        temperature=0.3,
        timeout=timeout,
    )


def _call_ollama_legacy(model: str, prompt: str, timeout: int = 90) -> str:
    """Single-shot Ollama call. Returns raw text response."""
    url = "http://localhost:11434/api/chat"
    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system",
             "content": "You are a financial analyst. Reply concisely."},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {"temperature": 0.3, "num_predict": 128},
    }).encode("utf-8")
    req = urllib.request.Request(url, data=payload,
                                  headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read().decode("utf-8"))
            return str(data.get("message", {}).get("content", "")).strip()
    except (urllib.error.URLError, TimeoutError, Exception) as e:
        return f"ERROR: {e}"


def _parse_direction(response: str) -> str:
    """Extract '+' or '-' from LLM response. Defaults to '+' on uncertain parse."""
    response = response.strip()
    # Look for + or - in first 30 chars; check both ascii and full-width
    head = response[:30]
    if "+" in head or "正" in head or "上涨" in head or "positive" in head.lower():
        return "+"
    if "-" in head or "−" in head or "负" in head or "下跌" in head or "negative" in head.lower():
        return "-"
    # Fallback: count + vs - in whole response
    pos_count = response.count("+") + response.count("正") + response.count("上涨")
    neg_count = response.count("-") + response.count("−") + response.count("负") + response.count("下跌")
    return "+" if pos_count >= neg_count else "-"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                       formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv-target",
                        default="data/processed/car_results.csv",
                        help="Target CAR archive (window=5 rows used)")
    parser.add_argument("--out", default=None,
                        help="Output CSV path (default: auto with timestamp)")
    parser.add_argument("--retries", type=int, default=2,
                        help="Retries per scenario on LLM call failure")
    args = parser.parse_args()

    model = get_llm_config()["model"]
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = args.out or f"data/processed/baseline_llm_only_{model.replace(':','_').replace('/','_')}_{ts}.csv"
    log_path = f"data/results/baseline_llm_only_{model.replace(':','_').replace('/','_')}_{ts}.txt"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    # Tee logging
    class Tee:
        def __init__(self, p):
            self.terminal = sys.stdout
            self.file = open(p, "w", encoding="utf-8")
        def write(self, m):
            self.terminal.write(m); self.file.write(m); self.file.flush()
        def flush(self):
            self.terminal.flush(); self.file.flush()
        def isatty(self):
            return self.terminal.isatty()
    tee = Tee(log_path); sys.stdout = tee

    print(f"=" * 72)
    print(f"LLM-Only Baseline (model: {model}, ts: {ts})")
    print(f"=" * 72)

    df = pd.read_csv(args.csv_target)
    df = df[df["window"] == 5].copy()
    df["industry_code"] = df["industry_code"].astype(str)
    df["event_date"]    = df["event_date"].astype(str).str[:10]

    print(f"Total events: {len(df)}\n")

    rows = []
    correct = 0
    for idx, row in df.iterrows():
        et = row["event_type"]
        ic = str(row["industry_code"])
        dt = str(row["event_date"])[:10]
        event_name = row["event_name"]
        real_car = float(row["CAR"])
        real_dir = "+" if real_car >= 0 else "-"

        ind_name = INDUSTRY_NAMES.get(ic, ic)
        prompt = (
            f"以下是中国A股市场的政策/宏观事件。\n"
            f"事件日期：{dt}\n"
            f"受影响行业：{ind_name}（申万一级行业代码 {ic}）\n"
            f"事件描述：{event_name}\n\n"
            f"问题：该行业在事件后5个交易日的累积异常收益（CAR）"
            f"相对于沪深300基准，是为正还是为负？\n"
            f"严格输出一个字符：+ 或 -"
        )

        pred_dir = ""
        raw = ""
        for attempt in range(args.retries + 1):
            raw = _call_llm(model, prompt)
            if not raw.startswith("ERROR:"):
                pred_dir = _parse_direction(raw)
                break
            print(f"  retry {attempt+1}/{args.retries+1}: {raw[:60]}")

        if not pred_dir:
            pred_dir = "+"  # default if all retries fail (will likely be wrong)

        is_correct = (pred_dir == real_dir)
        correct += int(is_correct)
        rows.append({
            "event_date":     dt,
            "industry_code":  ic,
            "event_type":     et,
            "event_name":     event_name[:50],
            "real_CAR":       real_car,
            "real_dir":       real_dir,
            "pred_dir":       pred_dir,
            "dir_correct":    is_correct,
            "llm_raw_response": raw[:120],
        })
        marker = "✅" if is_correct else "❌"
        print(f"[{idx+1:>2}/{len(df)}] {dt} {ic} {et:<20} "
              f"real={real_dir}({real_car:+.4f}) pred={pred_dir} {marker}")

    res_df = pd.DataFrame(rows)
    res_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"\n" + "=" * 72)
    print(f"LLM-Only Baseline Summary")
    print(f"=" * 72)
    acc = correct / len(df)
    print(f"CAR sign accuracy: {acc:.2%} ({correct}/{len(df)})")
    print(f"\nBy event type:")
    g = res_df.groupby("event_type")["dir_correct"].agg(["mean", "count"])
    g.columns = ["accuracy", "n"]
    g["accuracy"] = g["accuracy"].map(lambda x: f"{x:.2%}")
    print(g.to_string())
    print(f"\nBy industry (n >= 3):")
    g2 = res_df.groupby("industry_code")["dir_correct"].agg(["mean", "count"])
    g2.columns = ["accuracy", "n"]
    g2 = g2[g2["n"] >= 3].sort_values("n", ascending=False)
    g2["accuracy"] = g2["accuracy"].map(lambda x: f"{x:.2%}")
    print(g2.to_string())

    print(f"\nCSV: {out_path}")
    print(f"Log: {log_path}")
    sys.stdout = tee.terminal
    return 0


if __name__ == "__main__":
    sys.exit(main())
