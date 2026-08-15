"""
Strong non-LLM baselines under the temporal firewall (reviewer request #6).
=========================================================================

Reviewer #6: "Baselines are too weak (only LLM direct prompting). Add at least:
majority class / sign prior, Logistic/LightGBM, pure price time-series, simple
text classifier -- otherwise we cannot tell whether the gain comes from the
complex system or a trivial threshold / class prior."

All baselines are evaluated WALK-FORWARD (expanding window, strictly pre-event
training) so they are directly comparable to the firewalled system (60.5%) and
the data-access-clean LLM-only baseline (59.5%). No baseline ever sees the target
event's own outcome or any post-t information.

Baselines
  B1 majority-class        : predict the majority CAR-sign among events dated < t
  B2 event-type sign prior : predict majority sign of the SAME event_type, < t
  B3 price-momentum rule   : parameter-free; sign of pre-t industry-vs-market momentum
  B4 logistic (price feats): LogReg on pre-t price features, expanding walk-forward
  B5 lightgbm (price feats): GBDT on the same features, expanding walk-forward
  B6 text classifier       : char n-gram TF-IDF on event text + LogReg, walk-forward
  --pure-TS (ts_pre_signal): the Informer/Kalman time-series head alone (from base CSV)
  --LLM-only               : direct prompting (from baseline_llm_only CSV)

Every row: overall / positive-class / negative-class accuracy + n.

CLI:  python scripts/run_strong_baselines.py
"""
from __future__ import annotations
import os, sys, io, json, datetime

if sys.platform == "win32" and hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(SCRIPT_DIR)
os.chdir(ROOT_DIR)

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

RNG = 42


# ─── Data loading ──────────────────────────────────────────────────────
def _read_prices():
    """Load HS300 benchmark and Shenwan industry daily closes (GBK-safe).
    Returns (mkt_close: Series indexed by date, ind_close: DataFrame date x code)."""
    hs = pd.read_csv("data/raw/hs300.csv")
    hs.columns = ["date", "open", "high", "low", "close", "vol"]
    hs["date"] = pd.to_datetime(hs["date"])
    mkt = hs.set_index("date")["close"].sort_index()

    ind = pd.read_csv("data/raw/industry_daily.csv")
    ind.columns = ["code", "date", "close", "open", "high", "low", "vol",
                   "amount", "industry_code", "industry_name"]
    ind["date"] = pd.to_datetime(ind["date"])
    ind["industry_code"] = ind["industry_code"].astype(str)
    close = ind.pivot_table(index="date", columns="industry_code",
                            values="close", aggfunc="last").sort_index()
    volume = ind.pivot_table(index="date", columns="industry_code",
                             values="vol", aggfunc="last").sort_index()
    return mkt, close, volume


def _feat_row(event_dt, code, mkt, close, volume):
    """Pre-event price features for (event_date, industry). STRICTLY < t.
    Returns dict of features, or None if insufficient history."""
    t = pd.Timestamp(event_dt)
    if code not in close.columns:
        return None
    ic = close[code].loc[close.index < t].dropna()
    mc = mkt.loc[mkt.index < t].dropna()
    vc = volume[code].loc[volume.index < t].dropna() if code in volume.columns else pd.Series(dtype=float)
    if len(ic) < 25 or len(mc) < 25:
        return None

    def ret(s, n):
        return float(s.iloc[-1] / s.iloc[-1 - n] - 1.0) if len(s) > n else 0.0

    ind_r5, ind_r20 = ret(ic, 5), ret(ic, 20)
    mkt_r5, mkt_r20 = ret(mc, 5), ret(mc, 20)
    ind_logret = np.diff(np.log(ic.values[-21:])) if len(ic) >= 21 else np.array([0.0])
    vol20 = float(np.std(ind_logret)) if ind_logret.size else 0.0
    # 60d position of price within its own range (mean-reversion proxy)
    win = ic.values[-60:] if len(ic) >= 60 else ic.values
    rng = (win.max() - win.min())
    pos60 = float((ic.iloc[-1] - win.min()) / rng) if rng > 0 else 0.5
    # volume trend: last-5 mean vs last-20 mean
    vtrend = 0.0
    if len(vc) >= 20:
        m5, m20 = float(vc.iloc[-5:].mean()), float(vc.iloc[-20:].mean())
        vtrend = (m5 / m20 - 1.0) if m20 > 0 else 0.0
    return {
        "ind_r5": ind_r5, "ind_r20": ind_r20,
        "mkt_r5": mkt_r5, "mkt_r20": mkt_r20,
        "rel_r5": ind_r5 - mkt_r5, "rel_r20": ind_r20 - mkt_r20,
        "vol20": vol20, "pos60": pos60, "vtrend": vtrend,
    }


FEAT_COLS = ["ind_r5", "ind_r20", "mkt_r5", "mkt_r20",
             "rel_r5", "rel_r20", "vol20", "pos60", "vtrend"]


# ─── Scoring ───────────────────────────────────────────────────────────
def score(y_true, y_pred):
    """y_true / y_pred are arrays of +1 / -1. Returns overall/pos/neg acc + n."""
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    overall = float((y_true == y_pred).mean())
    pos = y_true == 1
    neg = y_true == -1
    pos_acc = float((y_pred[pos] == 1).mean()) if pos.any() else float("nan")
    neg_acc = float((y_pred[neg] == -1).mean()) if neg.any() else float("nan")
    return {"overall": overall, "pos": pos_acc, "neg": neg_acc,
            "n": int(len(y_true)), "n_pos": int(pos.sum()), "n_neg": int(neg.sum())}


def _fmt(name, s):
    def pc(x):
        return "  n/a " if x != x else f"{x*100:5.2f}%"
    return (f"  {name:<26} overall={pc(s['overall'])}  "
            f"pos={pc(s['pos'])}  neg={pc(s['neg'])}  "
            f"(n={s['n']}, +{s['n_pos']}/-{s['n_neg']})")


# ─── Build spine ───────────────────────────────────────────────────────
def build_spine(base_csv, llm_csv):
    """One row per (event_date, industry_code): real dir, event_type,
    pure-TS signal, LLM-only pred. Sorted by event_date (walk-forward order)."""
    car = pd.read_csv(os.environ.get("CAR_CSV", "data/processed/car_results.csv"))
    car = car[car["window"] == 5].copy()
    car["industry_code"] = car["industry_code"].astype(str)
    car["event_date"] = pd.to_datetime(car["event_date"])
    car["y"] = np.where(car["CAR"] >= 0, 1, -1)
    spine = car[["event_date", "event_type", "event_name",
                 "industry_code", "CAR", "y"]].copy()

    base = pd.read_csv(base_csv)
    base["industry_code"] = base["industry_code"].astype(str)
    base["event_date"] = pd.to_datetime(base["event_date"])
    ts_map = base.set_index(["event_date", "industry_code"])["ts_pre_signal"]
    spine["ts_sig"] = spine.set_index(["event_date", "industry_code"]).index.map(ts_map)

    if llm_csv and os.path.exists(llm_csv):
        llm = pd.read_csv(llm_csv)
        llm["industry_code"] = llm["industry_code"].astype(str)
        llm["event_date"] = pd.to_datetime(llm["event_date"])
        lm = llm.set_index(["event_date", "industry_code"])["pred_dir"]
        spine["llm_pred"] = spine.set_index(["event_date", "industry_code"]).index.map(lm)
    else:
        spine["llm_pred"] = np.nan

    spine = spine.sort_values(["event_date", "industry_code"]).reset_index(drop=True)
    return spine


def run(spine):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import make_pipeline
    import lightgbm as lgb

    mkt, close, volume = _read_prices()

    # Pre-compute price features per row (strictly pre-t)
    feats = []
    for _, r in spine.iterrows():
        f = _feat_row(r["event_date"], r["industry_code"], mkt, close, volume)
        feats.append(f if f else {c: 0.0 for c in FEAT_COLS})
    feat_df = pd.DataFrame(feats, columns=FEAT_COLS)

    events = spine["event_date"].values
    y = spine["y"].values
    et = spine["event_type"].values
    txt = spine["event_name"].astype(str).values
    n = len(spine)

    # Global majority sign of the whole benchmark (for cold-start fallback)
    global_maj = 1 if (y == 1).mean() >= 0.5 else -1

    preds = {k: np.zeros(n, dtype=int) for k in
             ["B1_majority", "B2_type_prior", "B3_momentum",
              "B4_logistic", "B5_lightgbm", "B6_text"]}

    for i in range(n):
        t = events[i]
        past = events < t                     # strict walk-forward mask
        yp = y[past]

        # B1 majority class among past events
        if yp.size == 0:
            preds["B1_majority"][i] = global_maj
        else:
            preds["B1_majority"][i] = 1 if (yp == 1).mean() >= 0.5 else -1

        # B2 event-type sign prior (past events of same type)
        same = past & (et == et[i])
        ys = y[same]
        if ys.size >= 3:
            preds["B2_type_prior"][i] = 1 if (ys == 1).mean() >= 0.5 else -1
        else:
            preds["B2_type_prior"][i] = preds["B1_majority"][i]

        # B3 parameter-free momentum: sign of relative 20d momentum
        rel = feat_df.iloc[i]["rel_r20"]
        preds["B3_momentum"][i] = 1 if rel >= 0 else -1

        # B4/B5 trained on past rows with usable features
        Xtr = feat_df[past].values
        ytr = yp
        Xte = feat_df.iloc[[i]].values
        if ytr.size >= 20 and len(np.unique(ytr)) == 2:
            try:
                sc = StandardScaler().fit(Xtr)
                lr = LogisticRegression(max_iter=1000, C=1.0)
                lr.fit(sc.transform(Xtr), ytr)
                preds["B4_logistic"][i] = int(lr.predict(sc.transform(Xte))[0])
            except Exception:
                preds["B4_logistic"][i] = preds["B1_majority"][i]
            try:
                gb = lgb.LGBMClassifier(n_estimators=120, num_leaves=15,
                                        learning_rate=0.05, min_child_samples=5,
                                        subsample=0.9, colsample_bytree=0.9,
                                        random_state=RNG, verbose=-1)
                gb.fit(Xtr, ytr)
                preds["B5_lightgbm"][i] = int(gb.predict(Xte)[0])
            except Exception:
                preds["B5_lightgbm"][i] = preds["B1_majority"][i]
        else:
            preds["B4_logistic"][i] = preds["B1_majority"][i]
            preds["B5_lightgbm"][i] = preds["B1_majority"][i]

        # B6 text classifier: char n-gram TF-IDF + LogReg on past event text
        if ytr.size >= 20 and len(np.unique(ytr)) == 2:
            try:
                clf = make_pipeline(
                    TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4),
                                    min_df=1, max_features=3000),
                    LogisticRegression(max_iter=1000, C=1.0))
                clf.fit(txt[past], ytr)
                preds["B6_text"][i] = int(clf.predict([txt[i]])[0])
            except Exception:
                preds["B6_text"][i] = preds["B1_majority"][i]
        else:
            preds["B6_text"][i] = preds["B1_majority"][i]

    return feat_df, preds, y


def main():
    # BASE_CSV: the walk-forward base run to pull ts_pre_signal/llm_signal from.
    # Override for the expanded 96-event benchmark (basex96 run).
    base_csv = os.environ.get(
        "BASE_CSV",
        "data/processed/all_experiment_results_qwen2.5_wf_base_20260717_163837.csv")
    # find latest LLM-only baseline CSV
    import glob
    llm_cands = sorted(glob.glob("data/processed/baseline_llm_only_*.csv"))
    llm_csv = llm_cands[-1] if llm_cands else None

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"data/results/strong_baselines_{ts}.txt"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    class Tee:
        def __init__(self, p):
            self.t = sys.stdout; self.f = open(p, "w", encoding="utf-8")
        def write(self, m): self.t.write(m); self.f.write(m); self.f.flush()
        def flush(self): self.t.flush(); self.f.flush()
        def isatty(self): return self.t.isatty()
    tee = Tee(log_path); sys.stdout = tee

    print("=" * 74)
    print("Strong non-LLM baselines under the temporal firewall (walk-forward)")
    print("=" * 74)
    print(f"base CSV : {base_csv}")
    print(f"LLM CSV  : {llm_csv}\n")

    spine = build_spine(base_csv, llm_csv)
    feat_df, preds, y = run(spine)

    results = {}
    print("Trained / rule baselines (expanding walk-forward, strictly pre-t):")
    for k in ["B1_majority", "B2_type_prior", "B3_momentum",
              "B4_logistic", "B5_lightgbm", "B6_text"]:
        s = score(y, preds[k]); results[k] = s
        print(_fmt(k, s))

    # Reference rows from existing artifacts
    print("\nReference systems (from logged artifacts):")
    ts_sig = spine["ts_sig"].map({"+": 1, "-": -1})
    m = ts_sig.notna().values
    if m.any():
        s = score(y[m], ts_sig[m].astype(int).values); results["pure_TS"] = s
        print(_fmt("pure-TS (Informer/KF)", s))
    llm = spine["llm_pred"].map({"+": 1, "-": -1})
    m2 = llm.notna().values
    if m2.any():
        s = score(y[m2], llm[m2].astype(int).values); results["LLM_only"] = s
        print(_fmt("LLM-only (direct prompt)", s))

    print("\nAnchors (from keyData_v4): firewalled base 3-seed 60.52%±0.28 "
          "(maj-vote 61.31%); pos~74.4% neg 47.29%.")

    # Save machine-readable summary
    out = {"generated": ts, "base_csv": base_csv, "llm_csv": llm_csv,
           "n": int(len(y)), "results": results}
    json_path = f"data/processed/strong_baselines_{ts}.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)

    # LaTeX-ready table snippet
    def pc(x): return "--" if x != x else f"{x*100:.2f}"
    order = [("B1_majority", "Majority class"),
             ("B2_type_prior", "Event-type sign prior"),
             ("B3_momentum", "Price-momentum rule"),
             ("B6_text", "Text classifier (TF-IDF)"),
             ("B4_logistic", "Logistic (price feats)"),
             ("B5_lightgbm", "LightGBM (price feats)"),
             ("pure_TS", "Time-series only (Informer/KF)"),
             ("LLM_only", "LLM-only (direct prompt)")]
    print("\n" + "=" * 74 + "\nLaTeX rows (overall / pos / neg):\n")
    for k, lab in order:
        if k in results:
            s = results[k]
            print(f"{lab:<32} & ${pc(s['overall'])}$ & ${pc(s['pos'])}$ & ${pc(s['neg'])}$ \\\\")

    print(f"\nJSON : {json_path}")
    print(f"Log  : {log_path}")
    sys.stdout = tee.t
    return 0


if __name__ == "__main__":
    sys.exit(main())
