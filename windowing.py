"""
事件窗口与历史样本切片工具。

这个模块只依赖 pandas / numpy，方便在没有 torch 的环境下单独做
时间对齐检查，避免事件实验里出现未来信息泄漏。
"""
from __future__ import annotations

import pandas as pd


FEATURE_COLS = [
    "ret",
    "vol_chg",
    "volatility",
    "mkt_ret",
    "price_ma5",
    "price_ma20",
]


def build_feature_frame(industry_code: str) -> pd.DataFrame:
    """
    为单个行业构造完整特征表。

    注意：
    - 先在全量历史上计算 pct_change / rolling 特征，再统一 dropna
    - 绝不能先截事件窗口再做 rolling，否则会破坏输入/目标的时间边界
    """
    ind_df = pd.read_csv("data/raw/industry_daily.csv")
    ind_df["日期"] = pd.to_datetime(ind_df["日期"])
    ind_df = ind_df[ind_df["industry_code"].astype(str) == str(industry_code)]
    ind_df = ind_df.sort_values("日期").set_index("日期")

    hs300 = pd.read_csv("data/raw/hs300.csv")
    hs300["日期"] = pd.to_datetime(hs300["日期"])
    hs300 = hs300.sort_values("日期").set_index("日期")

    df = ind_df[["收盘", "成交量"]].copy()
    df["mkt_close"] = hs300["收盘"]
    df = df.dropna()

    df["ret"] = df["收盘"].pct_change()
    df["vol_chg"] = df["成交量"].pct_change()
    df["volatility"] = df["ret"].rolling(5).std()
    df["mkt_ret"] = df["mkt_close"].pct_change()
    df["price_ma5"] = df["收盘"].rolling(5).mean() / df["收盘"] - 1
    df["price_ma20"] = df["收盘"].rolling(20).mean() / df["收盘"] - 1

    feature_df = df[FEATURE_COLS].dropna().astype("float32")
    return feature_df


def slice_event_window(
    feature_df: pd.DataFrame,
    event_date: str,
    seq_len: int,
    pred_len: int,
) -> pd.DataFrame:
    """
    返回 [输入窗口; 预测窗口] 拼接后的事件样本。

    规则：
    - 输入窗口：事件日(含首个 >= event_date 的交易日)之前的 seq_len 条特征
    - 预测窗口：从事件交易日起向后的 pred_len 条特征
    """
    event_dt = pd.to_datetime(event_date)
    anchor_idx = int(feature_df.index.searchsorted(event_dt))
    if anchor_idx >= len(feature_df):
        raise ValueError(f"事件日 {event_date} 之后无可用特征数据")

    start_idx = anchor_idx - seq_len
    end_idx = anchor_idx + pred_len

    if start_idx < 0:
        raise ValueError(
            f"事件日前特征数据不足 {seq_len} 个交易日，"
            f"实际只有 {anchor_idx} 个"
        )
    if end_idx > len(feature_df):
        raise ValueError(f"事件日后特征数据不足 {pred_len} 个交易日")

    window = feature_df.iloc[start_idx:end_idx].copy()
    expected_len = seq_len + pred_len
    if len(window) != expected_len:
        raise ValueError(
            f"事件窗口长度异常：期望 {expected_len}，实际 {len(window)}"
        )
    return window


def slice_history_before_event(feature_df: pd.DataFrame, event_date: str) -> pd.DataFrame:
    """
    返回事件日前的全部可用历史特征，严格排除事件日及其后数据。
    """
    event_dt = pd.to_datetime(event_date)
    history_df = feature_df[feature_df.index < event_dt].copy()
    if history_df.empty:
        raise ValueError(f"事件日 {event_date} 之前无可用历史特征")
    return history_df


def compute_window_scaler(window: pd.DataFrame, seq_len: int) -> dict:
    """Compute local z-score statistics from the input segment of an event window.

    Keeps the scaling parameters local to one event prediction instead of
    relying on a mutable global ``data/processed/scaler.json`` file. This is
    important for batch/walk-forward runs: concurrent events must not share
    scaler state.
    """
    if seq_len <= 0 or len(window) < seq_len:
        raise ValueError(
            f"window length {len(window)} is smaller than seq_len {seq_len}"
        )
    train_arr = window.iloc[:seq_len].to_numpy(dtype="float32")
    mean = train_arr.mean(axis=0)
    std = train_arr.std(axis=0) + 1e-8
    return {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "feature_cols": FEATURE_COLS,
    }
