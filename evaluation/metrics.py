"""
统一三维评测体系
维度 1：预测效果（误差/相关性/方向准确率/区间校准 + 关键时点方向）
维度 2：解释质量（链条一致性/证据充分性/图一致性）
维度 3：稳定性与回测（Sharpe/MDD/覆盖率稳定性）
"""
from __future__ import annotations
import json
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple


# ──────────────────────────────────────────────
# 维度 1：预测效果
# ──────────────────────────────────────────────
@dataclass
class PredictionMetrics:
    MAE: float = 0.0
    MSE: float = 0.0
    RMSE: float = 0.0
    MAPE: float = 0.0
    DirectionAcc: float = 0.0
    Coverage_80: float = 0.0
    Coverage_95: float = 0.0
    IntervalWidth_95: float = 0.0

def compute_prediction_metrics(
    preds: np.ndarray,
    actuals: np.ndarray,
    lower_80: Optional[np.ndarray] = None,
    upper_80: Optional[np.ndarray] = None,
    lower_95: Optional[np.ndarray] = None,
    upper_95: Optional[np.ndarray] = None,
) -> PredictionMetrics:
    mae = float(np.mean(np.abs(preds - actuals)))
    mse = float(np.mean((preds - actuals) ** 2))
    rmse = float(np.sqrt(mse))
    mape = float(np.mean(np.abs((preds - actuals) / (np.abs(actuals) + 1e-8))))
    direction_acc = float(np.mean(np.sign(preds) == np.sign(actuals)))

    cov80 = cov95 = iw95 = 0.0
    if lower_80 is not None and upper_80 is not None:
        cov80 = float(np.mean((actuals >= lower_80) & (actuals <= upper_80)))
    if lower_95 is not None and upper_95 is not None:
        cov95 = float(np.mean((actuals >= lower_95) & (actuals <= upper_95)))
        iw95 = float(np.mean(upper_95 - lower_95))

    return PredictionMetrics(
        MAE=round(mae, 4), MSE=round(mse, 6), RMSE=round(rmse, 4),
        MAPE=round(mape, 4), DirectionAcc=round(direction_acc, 4),
        Coverage_80=round(cov80, 4), Coverage_95=round(cov95, 4),
        IntervalWidth_95=round(iw95, 4),
    )


# ──────────────────────────────────────────────
# 维度 1-B：关键时点累积收益方向
# 用途：替代"24 步逐日方向"这种接近随机的指标，
#      采用事件研究（event study）文献惯例的 T+k 窗口方向
# 口径：行业累计收益（复利）的符号，不扣市场基准
#      —— 预测阶段不知未来 HS300，不引入未来信息
# ──────────────────────────────────────────────
@dataclass
class KeyPointDirectionMetrics:
    """T+k 窗口累积收益方向命中情况（按行业累积收益符号）"""
    key_points: List[int]                         = field(default_factory=lambda: [1, 3, 5, 10])
    pred_cum: Dict[int, float]                    = field(default_factory=dict)
    actual_cum: Dict[int, float]                  = field(default_factory=dict)
    direction_correct: Dict[int, bool]            = field(default_factory=dict)

    @property
    def summary(self) -> Dict[str, float]:
        """扁平化，方便并入 JSON 报告"""
        flat: Dict[str, float] = {}
        for k in self.key_points:
            flat[f"CAR_pred_T{k}"]         = float(self.pred_cum.get(k, 0.0))
            flat[f"CAR_actual_T{k}"]       = float(self.actual_cum.get(k, 0.0))
            flat[f"CAR_dir_correct_T{k}"]  = bool(self.direction_correct.get(k, False))
        return flat


def compute_keypoint_direction_metrics(
    preds: np.ndarray,
    actuals: np.ndarray,
    key_points: Tuple[int, ...] = (1, 3, 5, 10),
) -> KeyPointDirectionMetrics:
    """
    对每个关键时点 k（T+k 窗口），计算前 k 步复利累积收益的符号是否一致。
    - preds/actuals：长度为 pred_len 的原始收益率序列（非对数）
    - 若 k > 可用步数，会自动 clip 到可用长度；如此时两侧长度 < 1，则该点标记 False
    """
    pred_cum_map:    Dict[int, float] = {}
    actual_cum_map:  Dict[int, float] = {}
    dir_correct_map: Dict[int, bool]  = {}

    preds_np   = np.asarray(preds,   dtype=float).ravel()
    actuals_np = np.asarray(actuals, dtype=float).ravel()
    usable_len = min(len(preds_np), len(actuals_np))

    for k in key_points:
        k_eff = min(int(k), usable_len)
        if k_eff <= 0:
            pred_cum_map[k]    = 0.0
            actual_cum_map[k]  = 0.0
            dir_correct_map[k] = False
            continue
        pred_cum   = float(np.prod(1.0 + preds_np[:k_eff])   - 1.0)
        actual_cum = float(np.prod(1.0 + actuals_np[:k_eff]) - 1.0)
        # np.sign(0) = 0；退化为用 >=0 判正，避免零值导致的 False Negative
        pred_sign   = 1 if pred_cum   >= 0 else -1
        actual_sign = 1 if actual_cum >= 0 else -1
        pred_cum_map[k]    = round(pred_cum,   6)
        actual_cum_map[k]  = round(actual_cum, 6)
        dir_correct_map[k] = bool(pred_sign == actual_sign)

    return KeyPointDirectionMetrics(
        key_points=list(key_points),
        pred_cum=pred_cum_map,
        actual_cum=actual_cum_map,
        direction_correct=dir_correct_map,
    )


# ──────────────────────────────────────────────
# 维度 2：解释质量
# ──────────────────────────────────────────────
@dataclass
class ExplanationMetrics:
    ConsistencyScore: float = 0.0      # 机理链步骤置信度均值
    EvidenceCoverage: float = 0.0      # 有引证的步骤占比
    GraphConsistency: float = 0.0      # 推理方向与图谱传导一致比例
    AvgChainLength: float = 0.0        # 平均链条长度

def compute_explanation_metrics(chains: List[Dict]) -> ExplanationMetrics:
    if not chains:
        return ExplanationMetrics()

    consistencies, coverages, chain_lens = [], [], []
    for chain in chains:
        steps = chain.get("chain", [])
        if not steps:
            continue
        confs = [s.get("confidence", 0.5) for s in steps]
        consistencies.append(np.mean(confs))
        has_evidence = [len(s.get("evidence_ids", [])) > 0 for s in steps]
        coverages.append(np.mean(has_evidence))
        chain_lens.append(len(steps))

    return ExplanationMetrics(
        ConsistencyScore=round(float(np.mean(consistencies)), 4),
        EvidenceCoverage=round(float(np.mean(coverages)), 4),
        GraphConsistency=0.0,   # 需图谱比对，此处 stub
        AvgChainLength=round(float(np.mean(chain_lens)), 2),
    )


# ──────────────────────────────────────────────
# 维度 3：回测与稳定性
# ──────────────────────────────────────────────
@dataclass
class BacktestMetrics:
    TotalReturn: float = 0.0
    AnnualizedReturn: float = 0.0
    SharpeRatio: float = 0.0
    MaxDrawdown: float = 0.0
    WinRate: float = 0.0
    ProfitFactor: float = 0.0

def compute_backtest_metrics(
    signal: np.ndarray,         # 预测信号（正→买，负→卖）
    returns: np.ndarray,        # 实际收益率序列
    cost: float = 0.001,
    slippage: float = 0.0005,
    risk_free: float = 0.02 / 252,
    sizing_mode: str = "sign",  # "sign" (±1 full) | "magnitude" | "magnitude_stoploss"
    typical_magnitude: float = 0.005,
    stop_loss_threshold: float = -0.03,
) -> BacktestMetrics:
    """
    Backtest metrics with multiple position-sizing modes:
      - "sign":             positions = ±1 (current default; what v1 uses)
      - "magnitude":        positions = clip(|signal| / typical_magnitude, 0, 1) * sign(signal)
                             — wider position when prediction confidence is high
      - "magnitude_stoploss": magnitude sizing + flat-out when cumulative
                              PnL drops below stop_loss_threshold (e.g., -3%)
    """
    if sizing_mode == "sign":
        positions = np.sign(signal)
    elif sizing_mode in ("magnitude", "magnitude_stoploss"):
        # Confidence-weighted sizing: |signal| / typical → [0, 1] capped
        confidence = np.clip(np.abs(signal) / max(typical_magnitude, 1e-8), 0.0, 1.0)
        positions = np.sign(signal) * confidence
    else:
        raise ValueError(f"unknown sizing_mode: {sizing_mode}")

    trades = np.diff(positions, prepend=0.0)
    transaction_cost = np.abs(trades) * (cost + slippage)
    pnl = positions * returns - transaction_cost

    # Stop-loss overlay: once cumulative PnL drops below threshold, flat the position
    if sizing_mode == "magnitude_stoploss":
        cum_pnl = np.cumsum(pnl)
        triggered = False
        for i in range(len(pnl)):
            if cum_pnl[i] < stop_loss_threshold and not triggered:
                triggered = True
                # Flat from i+1 onward (close at this step's price, no further P&L)
                positions[i+1:] = 0.0
                # Recompute pnl tail
                tail_trades = np.diff(positions[i:], prepend=positions[i-1] if i>0 else 0.0)
                pnl[i+1:] = -np.abs(tail_trades[1:]) * (cost + slippage)
                break
    cum_return = np.cumprod(1 + pnl) - 1
    total_ret = float(cum_return[-1]) if len(cum_return) > 0 else 0.0

    n = len(pnl)
    # 避免负数开小数次幂产生复数
    base = max(1 + total_ret, 1e-8)
    annual_ret = float(base ** (252 / max(n, 1)) - 1)

    excess = pnl - risk_free
    sharpe = float(np.mean(excess) / (np.std(excess) + 1e-8) * np.sqrt(252))

    # 最大回撤
    nav = np.cumprod(1 + pnl)
    roll_max = np.maximum.accumulate(nav)
    drawdown = (roll_max - nav) / (roll_max + 1e-8)
    mdd = float(np.clip(np.max(drawdown), 0, 1))

    win_rate = float(np.mean(pnl > 0))
    gains = pnl[pnl > 0].sum()
    losses = -pnl[pnl < 0].sum()
    profit_factor = float(gains / (losses + 1e-8))

    return BacktestMetrics(
        TotalReturn=round(total_ret, 4),
        AnnualizedReturn=round(annual_ret, 4),
        SharpeRatio=round(sharpe, 4),
        MaxDrawdown=round(mdd, 4),
        WinRate=round(win_rate, 4),
        ProfitFactor=round(profit_factor, 4),
    )


# ──────────────────────────────────────────────
# 统一评测报告
# ──────────────────────────────────────────────
@dataclass
class EvaluationReport:
    scenario: str
    prediction: PredictionMetrics               = field(default_factory=PredictionMetrics)
    keypoints: KeyPointDirectionMetrics         = field(default_factory=KeyPointDirectionMetrics)
    explanation: ExplanationMetrics             = field(default_factory=ExplanationMetrics)
    backtest: BacktestMetrics                   = field(default_factory=BacktestMetrics)

    def to_json(self) -> str:
        # dataclass asdict 会递归序列化，但 int 键会原样保留；为兼容 JSON，
        # 先把 keypoints 三个 dict 的 int 键转成 str。
        d = asdict(self)
        kp = d.get("keypoints", {})
        for fld in ("pred_cum", "actual_cum", "direction_correct"):
            if fld in kp and isinstance(kp[fld], dict):
                kp[fld] = {str(k): v for k, v in kp[fld].items()}
        return json.dumps(d, ensure_ascii=False, indent=2)

    def print_summary(self) -> None:
        print(f"\n{'='*50}")
        print(f"评测报告 | 场景：{self.scenario}")
        print(f"{'='*50}")
        print("[预测效果]")
        print(f"  MAE={self.prediction.MAE}  RMSE={self.prediction.RMSE}")
        print(f"  步内方向={self.prediction.DirectionAcc:.2%}（24 步逐日，接近随机为理论上限）")
        if self.keypoints.key_points:
            kp_str = "  ".join(
                f"T+{k}={'[Y]' if self.keypoints.direction_correct.get(k, False) else '[N]'}"
                f"(pred={self.keypoints.pred_cum.get(k, 0.0):+.4f} "
                f"actual={self.keypoints.actual_cum.get(k, 0.0):+.4f})"
                for k in self.keypoints.key_points
            )
            print(f"  关键时点方向 [{kp_str}]")
        print(f"  区间覆盖率 80%={self.prediction.Coverage_80:.2%} / 95%={self.prediction.Coverage_95:.2%}")
        print("[解释质量]")
        print(f"  链条一致性={self.explanation.ConsistencyScore:.3f}")
        print(f"  证据覆盖率={self.explanation.EvidenceCoverage:.2%}")
        print("[回测]")
        print(f"  年化收益={self.backtest.AnnualizedReturn:.2%}  Sharpe={self.backtest.SharpeRatio:.3f}")
        print(f"  最大回撤={self.backtest.MaxDrawdown:.2%}  胜率={self.backtest.WinRate:.2%}")
