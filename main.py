"""
多智能体金融分析系统 — 主入口
端到端闭环：Agent-RAG → RAT×KG 推理 → 双路时序预测 → 统一评测

消融实验控制变量：
  ABLATION_MODE = "none"   → 完整系统（默认）
  ABLATION_MODE = "A1"     → 去掉外生修正
  ABLATION_MODE = "A2"     → 去掉KF分支（门控权重=0，纯Informer）
  ABLATION_MODE = "A3"     → 去掉RAG（检索结果置空）
  ABLATION_MODE = "A4"     → A1+A2同时（无外生修正+纯Informer）
"""

from __future__ import annotations
import os
import json
import logging
import random
import numpy as np
import torch
import pandas as pd
from config import CONFIG
from windowing import (
    FEATURE_COLS,
    build_feature_frame,
    compute_window_scaler,
    slice_event_window,
    slice_history_before_event,
)

# ── 消融模式控制 ──────────────────────────────
# 修改此变量切换消融配置，无需改动其他代码
ABLATION_MODE = os.environ.get("ABLATION_MODE", "none")

# ── v4 时间防火墙开关 ──────────────────────────
# WALK_FORWARD=1 启用严格 walk-forward 时间隔离（Phase 0）：
#   L1 敏感度先验只用 event_date<t 的事件；L2 永不读方向标签；
#   L3/L4/L5 RAG 只检索 pub_date<t 的文档（默认语料 doc_library_events.json，
#   隔离 18 篇前瞻 mechanism 文档）。产出诚实基线。
# 默认关闭以保持与 v3 行为可复现对照。
WALK_FORWARD = os.environ.get("WALK_FORWARD", "0") == "1"
from firewall import WalkForwardContext
from contracts import TransmissionMap

# ── v4 Phase 3 传导先验开关 ────────────────────
# USE_IRF=1 启用 IRF 传导先验软混合（固定权重 λ，向 IRF 收缩方向信号）。
# 默认关：保持 Phase 0/1 行为可复现，且使 IRF 成为可消融的独立组件。
USE_IRF = os.environ.get("USE_IRF", "0") == "1"

# ── v4 Phase 2 基本面 skill 开关 ────────────────
# USE_FUNDAMENTALS=1 启用价格派生基本面代理（<t 隔离，写入 tm cell 的
# path/confidence，不改方向）。默认关，可消融。
USE_FUNDAMENTALS = os.environ.get("USE_FUNDAMENTALS", "0") == "1"

# ── v4 Phase 2 part3 真实估值 skill 开关 ────────
# USE_FUND_VALUATION=1 启用真实行业估值探针（PE/PB/股息率历史分位，<t 隔离，
# 写 tm cell path/confidence，不改方向）。默认关，可消融。
USE_FUND_VALUATION = os.environ.get("USE_FUND_VALUATION", "0") == "1"

# USE_DEBATE=1: bull/bear multi-round debate + judge (variant B, confidence-gated
# abstention). Judge confidence < DEBATE_ABSTAIN -> abstain (signal 0, defer to
# time-series). Selective prediction: commit only on high-consensus events.
# leak-safe (event text + firewalled evidence only). Default off, ablatable.
USE_DEBATE = os.environ.get("USE_DEBATE", "0") == "1"

# USE_RELDECOMP=1: relative-return decomposition intervention (Phase 7). Forces
# the LLM to answer in 3 steps (R_ind, R_mkt, sign(R_ind-R_mkt)) instead of the
# CAR sign directly, to test causally whether the absolute-vs-relative conflation
# drives the optimism bias. leak-safe (event text only). Default off.
USE_RELDECOMP = os.environ.get("USE_RELDECOMP", "0") == "1"

# CAR_CSV lets the whole pipeline point at the expanded (96-event) benchmark
# without editing code. Default = original 168-scenario file (byte-identical runs).
CAR_CSV = os.environ.get("CAR_CSV", "data/processed/car_results.csv")

SEED = int(os.environ.get("SEED", "42"))
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
print(f"🎲 SEED = {SEED}")

if not os.environ.get("ANTHROPIC_API_KEY"):
    _llm = os.environ.get("LLM_MODEL", "qwen2.5:32b")
    print(f"🤖 使用本地 Ollama 模型（{_llm}）运行")

if ABLATION_MODE != "none":
    print(f"🔬 消融模式：{ABLATION_MODE}")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# 板块异质性敏感性矩阵（外生修正放缩系数）
# ──────────────────────────────────────────────
# key   = (event_type, industry_code)
# value = multiplier，乘到 net_signal 上
#   · 1.0 = 保持事件方向与强度（默认）
#   · 0 < v < 1 = 同向衰减
#   · v = 0 = 压制外生修正（信号模糊或历史幅度接近零）
#   · v < 0 = 翻转（政策对该板块有反向短期效应）
#   · v > 1 = 放大（对流动性/政策最敏感的板块）
#
# 本矩阵由 scripts/calibrate_sensitivity_v2.py 自动生成（Tier M, n=168），
# 然后**所有负 cell clamp 到 0**（2026-05-22 Tier M Fix）：
#
# 层次贝叶斯部分池化（hierarchical shrinkage）将 local (e, i) 统计与
# event-type-pooled / industry-pooled 两个共享先验做加权融合：
#   s_{(e,i)} = w * s_local + (1 - w) * s_shared
#   w = min(1, n_{(e,i)} / n_full),  n_full = 10 （与 run_all_events_v2.py 一致）
#
# 为什么 clamp 到 0（论文方法学要点）：
# 原始 v2 标定基于 P_pos（CAR 符号分布）给负 cell 表示"历史 CAR 多数负向"。
# 在 n=88 时事件偏正向（monetary_policy 多数）该信号有效；但扩到 n=168
# 后 market_event/geopolitical 占比上升，这些事件 LLM 已经正确读出负向，
# 与负 sensitivity 双负相乘后 correction 翻正向，导致 sanity 从 59% 跌到
# 51.79%。clamp 后 sensitivity 仅作"放大器"（永不翻向），LLM 控方向，
# 与论文的"机理 Agent 控方向、时序+先验控量级"叙事一致。
#
# 注意：scripts/run_all_events_v2.py 在跑批时同样会做 clamp（见该文件
# 第 27-38 行），保持 production / smoke test 数值一致。
#
# 标定数据：data/processed/car_results.csv  (window=5, 168 rows)
# 重生成命令：python scripts/calibrate_sensitivity_v2.py --n-full 10
#            然后手工 clamp 负值（或由 run_all_events_v2.py 自动 clamp）
#
# 修订记录：
#   2026-04-24 第二轮：5 行一阶段自动标定（CAR 51.14% → 56.82%）
#   2026-04-28 第三轮：手工微调 801710/801180（CAR 56.82% → 64.77%；buggy）
#   2026-05-21 Tier M：n=88 → n=168, 25 → 50 events, v2 shrinkage, 38 cells
#   2026-05-22 Tier M Fix：clamp 33 个负 cell 到 0（5 个正 cell 保留）
SECTOR_SENSITIVITY: dict = {
    ('capital_market_policy'   , '801030'):  0.0,
    ('capital_market_policy'   , '801180'):  0.0,
    ('capital_market_policy'   , '801710'):  0.0,
    ('capital_market_policy'   , '801750'): +0.307,
    ('capital_market_policy'   , '801760'):  0.0,
    ('capital_market_policy'   , '801770'):  0.0,
    ('geopolitical'            , '801010'):  0.0,
    ('geopolitical'            , '801030'):  0.0,
    ('geopolitical'            , '801040'):  0.0,
    ('geopolitical'            , '801050'):  0.0,
    ('geopolitical'            , '801080'):  0.0,
    ('geopolitical'            , '801110'):  0.0,
    ('geopolitical'            , '801730'):  0.0,
    ('geopolitical'            , '801770'):  0.0,
    ('geopolitical'            , '801880'):  0.0,
    ('geopolitical'            , '801890'):  0.0,
    ('industry_policy'         , '801030'):  0.0,
    ('industry_policy'         , '801050'):  0.0,
    ('industry_policy'         , '801080'):  0.0,
    ('industry_policy'         , '801160'):  0.0,
    ('industry_policy'         , '801180'):  0.0,
    ('industry_policy'         , '801710'):  0.0,
    ('industry_policy'         , '801750'): +0.112,
    ('industry_policy'         , '801760'):  0.0,
    ('industry_policy'         , '801770'):  0.0,
    ('industry_policy'         , '801780'): +0.110,
    ('industry_policy'         , '801790'):  0.0,
    ('industry_policy'         , '801880'):  0.0,
    ('market_event'            , '801080'):  0.0,
    ('market_event'            , '801730'):  0.0,
    ('market_event'            , '801750'):  0.0,
    ('market_event'            , '801770'):  0.0,
    ('market_event'            , '801780'):  0.0,
    ('market_event'            , '801790'):  0.0,
    ('market_event'            , '801880'):  0.0,
    ('monetary_policy'         , '801180'):  0.0,
    ('monetary_policy'         , '801750'): +0.481,
    ('monetary_policy'         , '801790'): +0.500,
}


def _lookup_event_type(event_date: str) -> str:
    """从 car_results.csv 查 event_type（用于敏感性矩阵键）。查不到返回空串"""
    try:
        df = pd.read_csv(CAR_CSV,
                         usecols=["event_date", "event_type"])
        df["event_date"] = df["event_date"].astype(str).str[:10]
        match = df[df["event_date"] == str(event_date)[:10]]
        if not match.empty:
            return str(match.iloc[0]["event_type"])
    except Exception as e:
        logger.warning(f"  [event_type 查找失败] {e}")
    return ""


def load_full_history(industry_code: str, event_date: str | None = None) -> np.ndarray:
    """
    加载全量历史数据用于模型预训练
    返回全部交易日的特征数组
    """
    feature_df = build_feature_frame(industry_code)
    if event_date:
        feature_df = slice_history_before_event(feature_df, event_date)

    arr = feature_df.values.astype(np.float32)
    if len(arr) < CONFIG.informer.seq_len + CONFIG.informer.pred_len:
        raise ValueError(
            f"行业 {industry_code} 在事件日前历史样本不足，"
            f"无法满足 seq_len={CONFIG.informer.seq_len} 与 pred_len={CONFIG.informer.pred_len}"
        )

    split = int(len(arr) * 0.8)
    mean = arr[:split].mean(axis=0)
    std  = arr[:split].std(axis=0) + 1e-8
    arr  = (arr - mean) / std

    enc_in = CONFIG.informer.enc_in
    if arr.shape[1] < enc_in:
        pad = np.zeros((len(arr), enc_in - arr.shape[1]), dtype=np.float32)
        arr = np.concatenate([arr, pad], axis=1)

    if event_date:
        logger.info(
            f"[load_full_history] 行业={industry_code} 事件日前历史形状={arr.shape} "
            f"日期范围={feature_df.index[0].date()}→{feature_df.index[-1].date()}"
        )
    else:
        logger.info(
            f"[load_full_history] 行业={industry_code} 全量数据形状={arr.shape} "
            f"日期范围={feature_df.index[0].date()}→{feature_df.index[-1].date()}"
        )
    return arr


def load_real_data_with_scaler(industry_code: str, event_date: str):
    """Return ``(standardized_event_window, local_scaler)`` for one event.

    Unlike the legacy ``load_real_data``, this function does NOT write
    ``data/processed/scaler.json``. Walk-forward batch runners should use this
    version so concurrent events never share global scaler state.
    """
    cfg = CONFIG.informer
    feature_df = build_feature_frame(industry_code)
    window = slice_event_window(feature_df, event_date, cfg.seq_len, cfg.pred_len)
    scaler = compute_window_scaler(window, cfg.seq_len)

    mean = np.asarray(scaler["mean"], dtype=np.float32)
    std = np.asarray(scaler["std"], dtype=np.float32)
    arr = window.values.astype(np.float32)
    arr = (arr - mean) / std

    enc_in = cfg.enc_in
    if arr.shape[1] < enc_in:
        pad = np.zeros((len(arr), enc_in - arr.shape[1]), dtype=np.float32)
        arr = np.concatenate([arr, pad], axis=1)
    elif arr.shape[1] > enc_in:
        arr = arr[:, :enc_in]

    input_start = window.index[0].date()
    input_end = window.index[cfg.seq_len - 1].date()
    target_start = window.index[cfg.seq_len].date()
    target_end = window.index[-1].date()
    logger.info(
        f"[load_real_data_with_scaler] shape={arr.shape}, "
        f"input={cfg.seq_len} rows ({input_start}->{input_end}), "
        f"target={cfg.pred_len} rows ({target_start}->{target_end})"
    )
    return arr, scaler


def load_real_data(industry_code: str, event_date: str) -> np.ndarray:
    """
    以事件日为基准，取事件日前 seq_len 个交易日作为输入窗口
    取事件日后 pred_len 个交易日作为预测目标
    """
    cfg = CONFIG.informer
    feature_df = build_feature_frame(industry_code)
    window = slice_event_window(feature_df, event_date, cfg.seq_len, cfg.pred_len)
    arr = window.values.astype(np.float32)

    train_arr = arr[:cfg.seq_len]
    mean = train_arr.mean(axis=0)
    std  = train_arr.std(axis=0) + 1e-8
    arr  = (arr - mean) / std

    enc_in = cfg.enc_in
    if arr.shape[1] < enc_in:
        pad = np.zeros((len(arr), enc_in - arr.shape[1]), dtype=np.float32)
        arr = np.concatenate([arr, pad], axis=1)
    elif arr.shape[1] > enc_in:
        arr = arr[:, :enc_in]

    logger.info(f"[load_real_data] 事件窗口形状={arr.shape}，"
                f"输入段={cfg.seq_len}行，预测段={cfg.pred_len}行")

    input_start = window.index[0].date()
    input_end = window.index[cfg.seq_len - 1].date()
    target_start = window.index[cfg.seq_len].date()
    target_end = window.index[-1].date()
    logger.info(
        f"[load_real_data] 事件窗口形状={arr.shape}，"
        f"输入段={cfg.seq_len}行({input_start}→{input_end})，"
        f"预测段={cfg.pred_len}行({target_start}→{target_end})"
    )

    return arr


# ──────────────────────────────────────────────
# 系统初始化
# ──────────────────────────────────────────────
def build_system(api_key: str = None, use_mock_kg: bool = True, blind_mode: bool = False,
                 event_date: str = "", walk_forward: bool = False):
    from rag.knowledge_graph import KnowledgeGraph
    from rag.retrieval import Retriever
    from agents.mechanism_agent import MechanismAgent
    from agents.data_agent import DataAgent

    logger.info("初始化知识图谱…")
    kg = KnowledgeGraph.build_demo() if use_mock_kg else KnowledgeGraph(use_mock=False)

    logger.info("初始化检索器…")
    if walk_forward:
        # 默认语料切到 event-tied（排除 18 篇前瞻 mechanism 文档，堵 L4）；
        # 允许 DOC_LIBRARY_PATH 覆盖。
        import os as _os
        corpus = _os.environ.get("DOC_LIBRARY_PATH",
                                 "data/events/doc_library_events.json")
        retriever = Retriever.build_from_library(path=corpus, kg=kg)
        # 用防火墙包装：只暴露 pub_date<event_date 的文档（堵 L3/L4/L5）
        ctx = WalkForwardContext(event_date=event_date)
        retriever = ctx.wrap_retriever(retriever)
        _fw_msg = (f"  [WALK-FORWARD ON] 检索器已按 event_date<{event_date} 过滤，"
                   f"可见文档={len(retriever.visible)}/"
                   f"{len(retriever.visible)+len(retriever.hidden)}")
        logger.info(_fw_msg); print(_fw_msg)   # print 进 Tee
    else:
        retriever = Retriever.build_from_library(kg=kg)

    logger.info("初始化机理 Agent…")
    mechanism_agent = MechanismAgent(retriever, kg, api_key=api_key, blind_mode=blind_mode)

    logger.info("初始化数据 Agent…")
    data_agent = DataAgent(device="cuda")

    return mechanism_agent, data_agent, kg


# ──────────────────────────────────────────────
# 端到端分析流程
# ──────────────────────────────────────────────
def run_analysis(
    event: str,
    entity_names: list,
    task_type: str = "policy_event",
    time_series: np.ndarray = None,
    industry_code: str = "801750",
    event_date: str = "2024-09-24",
    event_direction: str = "",
    api_key: str = None,
    blind_mode: bool = False,
    walk_forward: bool = None,
) -> dict:
    """
    完整闭环分析：
    1. 机理 Agent → 机理链 + 外生因子
    2. 数据 Agent → 时序预测
    3. 残差监控 → 触发逆向取证（如需要）
    4. 评测报告

    event_direction: "+"/"-"，来自事件库标注，用于生产模式引导推理。
    blind_mode:      True = 盲测模式，屏蔽所有方向先验泄漏（用于评测）。
                     False = 生产模式，可使用外部方向信号（默认）。
    """
    from evaluation.metrics import (
        compute_prediction_metrics, compute_explanation_metrics,
        compute_backtest_metrics, compute_keypoint_direction_metrics,
        EvaluationReport
    )

    # ── v4 walk-forward：强制无标签、时间隔离 ──────
    if walk_forward is None:
        walk_forward = WALK_FORWARD
    # 2^3 firewall factorial (reviewer 5a): per-channel toggles. UNSET => follow
    # walk_forward, so an existing walk-forward run is byte-identical to before.
    # Set FW_L1/FW_L2/FW_RAG explicitly (0/1) only in the factorial runner.
    def _fw(name: str, default: bool) -> bool:
        v = os.environ.get(name)
        return default if v is None else (v == "1")
    _l1 = _fw("FW_L1", walk_forward)   # sensitivity prior only uses event_date<t
    _l2 = _fw("FW_L2", walk_forward)   # never read direction label
    _l3 = _fw("FW_RAG", walk_forward)  # retrieval restricted to pub_date<t
    _wf_sensitivity = None   # walk-forward 敏感度矩阵（只含 event_date<t 的事件）
    if _l2:
        # L2：永远不读方向标签，等价 blind 且断掉所有 fallback
        blind_mode = True
        event_direction = ""
    if _l1:
        # L1：敏感度先验只用严格早于 t 的事件（替换全档案 SECTOR_SENSITIVITY）
        _wf_ctx = WalkForwardContext(event_date=event_date)
        _wf_sensitivity = _wf_ctx.sensitivity_prior()
        _wf_msg = (f"  [FW L1 ON] 敏感度先验基于 {_wf_ctx.n_past_events()} "
                   f"个历史事件（<{event_date}），{len(_wf_sensitivity)} 个非平凡 cell")
        logger.info(_wf_msg); print(_wf_msg)   # print 进 Tee，确保全量日志可见
    if walk_forward:
        _fwmsg = f"  [FIREWALL] L1={int(_l1)} L2={int(_l2)} RAG={int(_l3)}"
        logger.info(_fwmsg); print(_fwmsg)

    # ── 自动查找事件方向（仅生产模式，盲测模式禁用以防泄漏）──
    if not blind_mode and not event_direction:
        try:
            ev_df = pd.read_csv("data/events/event_library.csv")
            ev_df["event_date"] = ev_df["event_date"].astype(str).str[:10]
            match = ev_df[ev_df["event_date"] == str(event_date)[:10]]
            if not match.empty:
                event_direction = str(match.iloc[0].get("direction", ""))
                logger.info(f"  [生产模式] 自动从事件库查找方向: {event_direction}")
        except Exception:
            pass
    elif blind_mode and event_direction:
        # 即使调用方传入了 event_direction，盲测模式也强制清空
        logger.info(f"  [盲测模式] 忽略传入的 event_direction='{event_direction}'，改为空")
        event_direction = ""

    # RAG firewall is gated by its own channel toggle (_l3), so the factorial can
    # unseal retrieval while keeping L1/L2 sealed (and vice versa).
    mechanism_agent, data_agent, kg = build_system(api_key=api_key, blind_mode=blind_mode,
                                                    event_date=event_date,
                                                    walk_forward=_l3)

    # ── Step 1: 机理推理 ────────────────────────
    logger.info(f"[Step 1] 机理推理：{event} (方向先验={event_direction or '无'})")
    chain = mechanism_agent.analyze(event, task_type=task_type, entity_names=entity_names,
                                    event_direction=event_direction)

    # ── Phase 1: 机理链 → 传导张量（行为保持）────────
    # TransmissionMap 是新的推理→数值契约；as_exog_factors() 精确复刻 v3
    # exog_factors，故下游标量口径不变。A3 消融仍置空。
    if ABLATION_MODE == "A3":
        transmission = TransmissionMap(event_date=str(event_date)[:10])
        exog_factors = {}
        logger.info("  [A3消融] 外生因子已置空（去掉RAG）")
    else:
        transmission = TransmissionMap.from_mechanism_chain(
            chain, industry_hint=str(industry_code),
            event_date=str(event_date)[:10],
            event_type=_lookup_event_type(event_date),
        )
        exog_factors = transmission.as_exog_factors()   # == v3 chain.to_exogenous_factors()

        # ── Phase 2: 基本面 skill（价格派生，<t 隔离；写 cell path/confidence，不改方向）──
        if USE_FUNDAMENTALS:
            from agents.fundamentals_agent import FundamentalsSkill
            _fund_pr = FundamentalsSkill().apply_to_map(
                transmission, str(industry_code), event_date)
            _fm = (f"  [基本面] pre_state={_fund_pr.pre_state:+.3f} "
                   f"conf={_fund_pr.confidence:.3f} n_hist={_fund_pr.n_hist} "
                   f"| {'; '.join(_fund_pr.layers) if _fund_pr.layers else '冷启动'}")
            logger.info(_fm); print(_fm)

        # ── Phase 2 part3: 真实行业估值 skill（PE/PB 历史分位，<t 隔离，不改方向）──
        if USE_FUND_VALUATION:
            from agents.fundamentals_agent import FundamentalsSkill
            _val = FundamentalsSkill().apply_valuation_to_map(
                transmission, str(industry_code), event_date)
            _vm = (f"  [估值] signal={_val.valuation_signal:+.3f} "
                   f"conf={_val.confidence:.3f} n_hist={_val.n_hist} "
                   f"| {_val.note or '冷启动/无数据'}")
            logger.info(_vm); print(_vm)

    _ef_msg = f"  外生因子：{exog_factors}"
    _conf_msg = f"  机理链置信度：{chain.overall_confidence:.3f}"
    logger.info(_ef_msg); print(_ef_msg)
    logger.info(_conf_msg); print(_conf_msg)

    # ── Step 2: 时序预测 ────────────────────────
    logger.info("[Step 2] 时序预测")
    if time_series is None:
        time_series, window_scaler = load_real_data_with_scaler(
            industry_code=industry_code,
            event_date=event_date,
        )
    else:
        window_scaler = None

    seq_len = CONFIG.informer.seq_len
    pred_len = CONFIG.informer.pred_len
    expected_len = seq_len + pred_len
    if len(time_series) < expected_len:
        raise ValueError(
            f"time_series 长度不足：期望至少 {expected_len}，实际 {len(time_series)}"
        )

    # time_series 的约定是 [事件日前输入窗口 ; 事件日起预测窗口]。
    # 这里必须显式切开，不能再用最后 seq_len 行，否则会把未来窗口重新喂进模型。
    x_input = time_series[:seq_len]
    actual_slice = time_series[seq_len:seq_len + pred_len]

    # ── 预事件漂移检测 ─────────────────────────────────
    # 若市场已在事件前提前涨/跌（定价充分），则压缩外生因子幅度
    # 原理：sell the news / buy the news 现象；LPR降息前市场常已上涨
    # 计算事件前 10 个交易日的累计标准化收益
    if window_scaler is not None:
        _ret_mean = float(window_scaler["mean"][0])
        _ret_std = float(window_scaler["std"][0])
    else:
        _ret_mean, _ret_std = 0.0, 1.0

    pre_window = 10
    pre_raw = x_input[-pre_window:, 0]          # 标准化收益率序列
    pre_real = pre_raw * _ret_std + _ret_mean    # 还原为真实收益率
    pre_cum_return = float(np.sum(pre_real))     # 累计收益率（前 10 日）

    # 预漂移强度：|前10日累计收益| / 典型事件幅度（约 3%）
    pre_drift_ratio = abs(pre_cum_return) / 0.03
    pre_drift_ratio = min(pre_drift_ratio, 1.0)   # 上限 1.0

    logger.info(f"  [预事件漂移] 前10日累计={pre_cum_return:.4f}  "
                f"压缩比={pre_drift_ratio:.2f}")

    full_history = load_full_history(
        industry_code=industry_code,
        event_date=event_date,
    )

    # 训练集：前 80%；校准集：后 20%（时间顺序，不打乱）
    split      = int(len(full_history) * 0.80)
    train_data = full_history[:split]
    calib_data = full_history[split:]

    data_agent.fit(train_data, epochs=100)

    # Conformal Prediction 区间校准（利用校准集计算校准因子）
    calib_result = data_agent.calibrate(calib_data)
    calib_msg = (
        f"  [区间校准] q80={calib_result['factor_80']:.3f} "
        f"q95={calib_result['factor_95']:.3f} "
        f"n_scores={calib_result['n_scores']}"
    )
    logger.info(calib_msg)
    print(calib_msg)   # 同步写入 Tee 日志文件（logger 仅输出到 stderr）

    # ── 外生信号计算 ────────────────────────────
    if exog_factors:
        vals = list(exog_factors.values())
        event_strength = float(max(vals, key=abs))  # 保留符号
        net_signal_inject = sum(vals)
    else:
        event_strength = 0.0
        net_signal_inject = 0.0

    # 事件强度兜底：当外生因子过弱时，从事件文本估算最低强度
    # 确保区间在事件窗口被适当放大，不受 LLM 因子质量影响
    if abs(event_strength) < 0.1:
        HIGH_IMPACT_TERMS = [
            "关税", "大幅", "一揽子", "双降", "降准", "组合拳",
            "黑色", "历史最大", "超预期", "暴跌", "大跌", "里程碑",
        ]
        hit_count = sum(1 for t in HIGH_IMPACT_TERMS if t in event)
        if hit_count >= 2:
            text_strength = 0.50
        elif hit_count == 1:
            text_strength = 0.30
        else:
            text_strength = 0.15   # 普通政策事件也给予基础强度
        # 保持方向与外生因子一致，若无外生因子则用文本推断方向
        from agents.mechanism_agent import MechanismAgent
        text_dir = MechanismAgent._infer_direction_from_text(event)
        sign = -1.0 if text_dir == "-" else 1.0
        event_strength = sign * text_strength
        logger.info(f"  [事件强度兜底] text_strength={event_strength:.3f} "
                    f"(hits={hit_count}, dir={text_dir or '?'})")

    # 如果外生因子方向不明确，且处于生产模式且有事件库标注方向，用标注方向作为兜底
    # 盲测模式下禁用此兜底，防止标签泄漏
    if not blind_mode and event_direction and abs(net_signal_inject) < 0.1:
        dir_sign = 1.0 if event_direction == "+" else -1.0
        net_signal_inject = dir_sign * 0.5
        event_strength = dir_sign * 0.5
        logger.info(f"  [生产模式] 外生因子弱，使用事件库方向兜底: {event_direction}")

    # ── 预事件漂移压缩 ──────────────────────────────────
    # 若市场已提前沿事件方向运动（sell the news），压缩注入幅度
    # 同向漂移（市场已涨且事件为正）→ 压缩；反向漂移 → 保持或略增
    drift_same_dir = (pre_cum_return > 0) == (event_strength > 0)
    if drift_same_dir and pre_drift_ratio > 0.3:
        # 压缩因子：漂移比例越大，压缩越强；最大压缩 60%
        dampen = 1.0 - min(pre_drift_ratio * 0.6, 0.60)
        orig_strength = event_strength
        event_strength     *= dampen
        net_signal_inject  *= dampen
        logger.info(f"  [预漂移压缩] {orig_strength:.3f} → {event_strength:.3f} "
                    f"(dampen={dampen:.2f}, 同向漂移={pre_cum_return:.4f})")

    # ── 滚动波动率比值：事件窗口 vs 历史均值 ─────────────
    # 若事件前夕波动率已明显放大，说明是高不确定性窗口，需额外扩区间
    # 这比固定 event_strength 扩展更客观，直接反映市场对事件的不确定性
    hist_vol = float(np.std(full_history[:, 0]) + 1e-8)    # 全量历史波动率
    recent_vol = float(np.std(x_input[-10:, 0]) + 1e-8)   # 近 10 日波动率
    vol_ratio = recent_vol / hist_vol                       # 比值：>1 说明近期更波动
    vol_ratio = float(np.clip(vol_ratio, 0.5, 4.0))        # 限幅

    logger.info(f"  [波动率比] hist_vol={hist_vol:.4f} "
                f"recent_vol={recent_vol:.4f} ratio={vol_ratio:.2f}")

    # 最终用于区间扩展的综合强度：取 event_strength 和 vol_ratio 调整值的较大者
    # 确保即使 LLM 外生因子弱，高波动也会触发扩区间
    vol_adjusted_strength = (vol_ratio - 1.0) * 0.5   # vol_ratio=2→strength=0.5
    effective_interval_strength = max(
        abs(event_strength),
        vol_adjusted_strength,
        0.15   # 最小基础强度，所有事件窗口均扩展
    )

    logger.info(f"  [有效区间强度] event={abs(event_strength):.3f} "
                f"vol_adj={vol_adjusted_strength:.3f} "
                f"final={effective_interval_strength:.3f}")
    x_inject = x_input.copy()
    inject_window = min(10, len(x_inject))
    for t in range(inject_window):
        # 越靠近事件日（最后一步），信号越强
        weight = (t + 1) / inject_window
        idx = -(inject_window - t)
        x_inject[idx, -1] = float(np.clip(net_signal_inject * weight, -2, 2))
        x_inject[idx, -2] = float(np.clip(event_strength * weight, -2, 2))

    # ── 纯 TS 预测（无 LLM 信号注入）：用于 disagreement-as-feature 分析 ──
    # 这是个独立的 forward pass，使用未注入 LLM 信号的原始 x_input
    # 用来记录 TS 模型在没有 mechanism agent 帮助下的方向预测
    try:
        pred_pure_ts = data_agent.predict(
            x_input,
            exog_factors=None,
            event_strength=0.0,
            market_volatility=vol_ratio,
            force_informer_only=(ABLATION_MODE in ("A2", "A4")),
        )
        ts_pre_pred5 = float(np.mean(pred_pure_ts.point_forecast[:5]))
        ts_pre_signal = "+" if ts_pre_pred5 > 0 else "-"
    except Exception as _e:
        logger.warning(f"  [pure-TS predict] 失败: {_e}")
        ts_pre_pred5 = 0.0
        ts_pre_signal = ""

    llm_signal = ("+" if net_signal_inject > 0
                  else "-" if net_signal_inject < 0 else "")
    disagreement_flag = bool(
        ts_pre_signal and llm_signal and ts_pre_signal != llm_signal
    )
    logger.info(f"  [disagreement] LLM={llm_signal} TS={ts_pre_signal} "
                f"disagree={disagreement_flag}")

    # ── A2消融：纯Informer，门控权重强制=0 ────────
    pred = data_agent.predict(
        x_inject,
        exog_factors=exog_factors,
        event_strength=effective_interval_strength,   # 综合强度：控制区间宽度
        market_volatility=vol_ratio,
        force_informer_only=(ABLATION_MODE in ("A2", "A4")),
    )

    logger.info(f"  {pred.to_summary()}")

    # ── 反标准化 ──────────────────────────────────
    if window_scaler is not None:
        ret_mean = float(window_scaler["mean"][0])
        ret_std = float(window_scaler["std"][0])
        pred_ret_real  = pred.point_forecast * ret_std + ret_mean
        pred_lo95_real = pred.interval_lower_95 * ret_std + ret_mean
        pred_hi95_real = pred.interval_upper_95 * ret_std + ret_mean
    else:
        logger.warning("  无局部 scaler，跳过反标准化")
        pred_ret_real  = pred.point_forecast
        pred_lo95_real = pred.interval_lower_95
        pred_hi95_real = pred.interval_upper_95
        ret_std, ret_mean = 1.0, 0.0

    # ── 外生因子修正（A1/A4消融时跳过）─────────────
    # 修复Bug2-5：更强的修正 + 方向冲突检测
    if ABLATION_MODE in ("A1", "A4"):
        logger.info("  [消融] 跳过外生因子修正")
    else:
        if exog_factors or event_direction:
            pos_sum    = sum(v for v in exog_factors.values() if v > 0)
            neg_sum    = sum(abs(v) for v in exog_factors.values() if v < 0)
            net_signal = pos_sum - neg_sum

            evt_type   = _lookup_event_type(event_date)

            # ── Phase 6: 多智能体辩论（variant B，USE_DEBATE 时启用）──────
            # bull/bear 多轮辩论 + judge 裁决，置换 net_signal 为 judge 信号；
            # judge 弃权（置信度低）时 net_signal=0（退回时序）。leak-safe：只喂
            # 事件描述文本（合法输入，与 _infer_direction_from_text 同源）。
            _debate_info = {}
            if USE_DEBATE:
                try:
                    from agents.debate_agent import run_debate
                    _dbg_model = os.environ.get("LLM_MODEL", "qwen2.5:32b")
                    _dsig, _debate_info = run_debate(
                        event=str(event), industry=str(industry_code),
                        evidence="", model=_dbg_model)
                    net_signal = _dsig      # judge verdict (0.0 if abstained)
                    _dm = (f"  [辩论] dir={_debate_info.get('direction','-')} "
                           f"conf={_debate_info.get('confidence')} "
                           f"abstain={_debate_info.get('abstained')} "
                           f"→net={net_signal:.3f}")
                    logger.info(_dm); print(_dm)
                except Exception as _de:
                    logger.warning(f"  [辩论] 失败，退回原 net_signal: {_de}")

            # ── Phase 7: 相对收益分解干预（USE_RELDECOMP 时启用）──────────
            # 显式三步(R_ind / R_mkt / 跑赢?)置换 net_signal，检验绝对-相对混淆
            # 是否为乐观偏置根因。与 debate 互斥使用。
            _reld_info = {}
            if USE_RELDECOMP:
                try:
                    from agents.reldecomp_agent import run_reldecomp
                    _rd_model = os.environ.get("LLM_MODEL", "qwen2.5:32b")
                    _rsig, _reld_info = run_reldecomp(
                        event=str(event), industry=str(industry_code),
                        evidence="", model=_rd_model)
                    if _reld_info.get("reldecomp") == "ok":
                        net_signal = _rsig
                    _rm = (f"  [相对分解] r_ind={_reld_info.get('r_ind')} "
                           f"r_mkt={_reld_info.get('r_mkt')} "
                           f"beats={_reld_info.get('beats_market')} →net={net_signal:.3f}")
                    logger.info(_rm); print(_rm)
                except Exception as _re2:
                    logger.warning(f"  [相对分解] 失败，退回原 net_signal: {_re2}")

            # ── Phase 3: IRF 传导先验软混合（USE_IRF 时启用；默认关，保持 Phase0/1 行为）──
            if USE_IRF:
                from agents.transmission_agent import TransmissionSkill
                _irf_skill = TransmissionSkill()
                net_signal, _irf_info = _irf_skill.blend_net_signal(
                    net_signal, evt_type, str(industry_code), event_date,
                    walk_forward=walk_forward)
                if _irf_info.get("irf") == "applied":
                    _im = (f"  [IRF软混合] dir={_irf_info['direction']} "
                           f"s_wf={_irf_info['strength_wf']} λ={_irf_info['lambda']} "
                           f"net {_irf_info['net_in']}→{_irf_info['net_out']}")
                    logger.info(_im); print(_im)

            # ── Shrinkage controls (reviewer #2): does a leak-free GENERIC
            # shrink match IRF's MAE gain? SHRINK_MODE is mutually exclusive with
            # USE_IRF (leak-free: uses no history, no direction).
            #   neutral  : net *= (1-λ)   (direction-free λ-shrink; IRF with s=0)
            #   fixed065 : net *= 0.65    (fixed global shrink in the 0.6-0.8 band)
            #   zero     : net  = 0       (pure time-series, no correction)
            _shrink = os.environ.get("SHRINK_MODE", "").strip().lower()
            if _shrink and not USE_IRF:
                _n_in = net_signal
                if _shrink == "neutral":
                    net_signal *= (1.0 - float(os.environ.get("IRF_LAMBDA", "0.35")))
                elif _shrink == "fixed065":
                    net_signal *= 0.65
                elif _shrink == "zero":
                    net_signal = 0.0
                _sm = f"  [SHRINK {_shrink}] net {_n_in:.3f}→{net_signal:.3f}"
                logger.info(_sm); print(_sm)

            # ── 板块异质性敏感性放缩 ─────────────────
            sens_key   = (evt_type, str(industry_code))
            # L1 sealed -> prior over event_date<t only; L1 unsealed -> the v3
            # full-archive matrix (leaky, for the factorial's dirty-prior cells).
            _sens_matrix = _wf_sensitivity if _l1 else SECTOR_SENSITIVITY
            sensitivity = _sens_matrix.get(sens_key, 1.0)
            if sensitivity != 1.0:
                msg = (f"  板块异质性: ({evt_type}, {industry_code}) "
                       f"sensitivity={sensitivity:+.2f}")
                logger.info(msg); print(msg)
            net_signal_adj = net_signal * sensitivity

            # 修正力度：小数收益率尺度。可通过环境变量 CORRECTION_SCALE 覆盖（用于 sweep）
            # 经验区间 0.02~0.03：小 → 方向翻转不足；大 → 量级 overshoot、MAE 恶化
            # 默认 0.02：冒烟 SCALE sweep 证实 0.02 已能翻转 2020-02-20/银行，且 MAE 最低（0.0387）
            # 真正接管方向判定的是 factor_direction 改基于 net_signal_adj 的修复，而非 SCALE 大小
            SCALE      = float(os.environ.get("CORRECTION_SCALE", "0.02"))
            correction = float(np.clip(net_signal_adj * SCALE, -0.08, 0.08))

            # ── 衰减曲线：两段式线性 + 下限 0.55（消除原 [1,0.95,...,0.7] 阶跃）──
            t_idx = np.arange(len(pred_ret_real))
            decay = np.clip(1.0 - 0.035 * t_idx, 0.55, 1.0)

            pred_ret_real = pred_ret_real + correction * decay

            msg = (f"  外生修正: SCALE={SCALE:.3f} "
                   f"net_signal={net_signal:.3f} "
                   f"adj={net_signal_adj:.3f} "
                   f"correction={correction:.4f} "
                   f"修正后前5步={pred_ret_real[:5].round(4)}")
            logger.info(msg); print(msg)

            # ── 修复Bug5：方向冲突检测与矫正（基于 net_signal_adj 以生效板块异质性）──
            if abs(net_signal_adj) > 0.3:
                factor_direction = "+" if net_signal_adj > 0 else "-"
            elif not blind_mode and event_direction:
                # 仅生产模式可使用事件库方向作为 fallback，盲测模式禁用
                factor_direction = event_direction
            else:
                factor_direction = ""

            if factor_direction:
                pred_mean_5 = float(np.mean(pred_ret_real[:5]))
                pred_dir = "+" if pred_mean_5 > 0 else "-"

                if pred_dir != factor_direction:
                    # 模型预测方向 vs 外生因子方向冲突！
                    # 施加额外的方向矫正（沿板块异质性方向）
                    conflict_strength = abs(net_signal_adj) if abs(net_signal_adj) > 0.3 else 0.5
                    dir_sign = 1.0 if factor_direction == "+" else -1.0
                    override = dir_sign * (abs(pred_mean_5) + conflict_strength * 0.005)
                    override = float(np.clip(override, -0.03, 0.03))
                    pred_ret_real = pred_ret_real + override * decay
                    msg = (f"  ⚠ 方向冲突！模型={pred_dir} vs 因子={factor_direction} "
                           f"→ 施加方向矫正 {override:+.4f}")
                    logger.warning(msg); print(msg)

    logger.info(f"  反标准化预测收益率（前5步）: {pred_ret_real[:5].round(4)}")
    logger.info(f"  95%区间（t+1）: [{pred_lo95_real[0]:.4f}, {pred_hi95_real[0]:.4f}]")

    # ── Step 3: 残差监控 ────────────────────────
    logger.info("[Step 3] 残差监控")
    actual_values      = actual_slice[:, 0]
    actual_values_real = actual_values * ret_std + ret_mean
    triggered_list     = []

    for t in range(min(5, len(actual_values_real))):
        triggered, reason = data_agent.update_residual(
            actual=float(actual_values_real[t]),
            predicted=float(pred_ret_real[t]),
        )
        if triggered:
            triggered_list.append(reason)

    if triggered_list:
        logger.warning(f"[Step 3] 触发逆向取证：{triggered_list[0]}")
        chain = mechanism_agent.reverse_retrieval(
            residual_info=data_agent.monitor.recent_residual_info(),
            time_window="近5步",
            affected_assets=entity_names,
            chain=chain,
        )

    # ── Step 4: 统一评测 ────────────────────────
    logger.info("[Step 4] 统一评测")
    pred_metrics = compute_prediction_metrics(
        preds=pred_ret_real,
        actuals=actual_values_real,
        lower_80=pred.interval_lower_80 * ret_std + ret_mean,
        upper_80=pred.interval_upper_80 * ret_std + ret_mean,
        lower_95=pred_lo95_real,
        upper_95=pred_hi95_real,
    )
    # 关键时点累积收益方向（替代 24 步逐日方向作为主方向指标）
    keypoint_metrics = compute_keypoint_direction_metrics(
        preds=pred_ret_real,
        actuals=actual_values_real,
        key_points=(1, 3, 5, 10),
    )
    exp_metrics = compute_explanation_metrics([chain.raw])
    bt_metrics  = compute_backtest_metrics(
        signal=pred_ret_real,
        returns=actual_values_real,
        sizing_mode="sign",
    )
    # ── Sharpe-improvement experiment (Plan 修复 Sharpe) ──────────────
    # Compute two additional Sharpe variants without altering the headline
    # backtest. Both are written into the result dict (and from there to CSV)
    # so the downstream backtest analysis can compare position-sizing strategies
    # without re-running the 88-event pipeline.
    bt_metrics_v2 = compute_backtest_metrics(
        signal=pred_ret_real,
        returns=actual_values_real,
        sizing_mode="magnitude",
        typical_magnitude=0.005,
    )
    bt_metrics_v3 = compute_backtest_metrics(
        signal=pred_ret_real,
        returns=actual_values_real,
        sizing_mode="magnitude_stoploss",
        typical_magnitude=0.005,
        stop_loss_threshold=-0.03,
    )
    # ── Trading-cost sensitivity analysis (Plan C, Stage 1.3) ──────
    # A-share double-side cost: stamp tax (sell-only 0.05%) + commission
    # (0.0003-0.001 each way) + slippage. Realistic round-trip total
    # ranges from 0.10% (institutional, low-slippage) to 0.30% (retail).
    # Report Sharpe under three cost levels to show robustness.
    bt_cost_010 = compute_backtest_metrics(
        signal=pred_ret_real, returns=actual_values_real,
        sizing_mode="sign", cost=0.0005, slippage=0.0005,  # 0.10% round-trip
    )
    bt_cost_020 = compute_backtest_metrics(
        signal=pred_ret_real, returns=actual_values_real,
        sizing_mode="sign", cost=0.0010, slippage=0.0010,  # 0.20% round-trip
    )
    bt_cost_030 = compute_backtest_metrics(
        signal=pred_ret_real, returns=actual_values_real,
        sizing_mode="sign", cost=0.0015, slippage=0.0015,  # 0.30% round-trip
    )

    report = EvaluationReport(
        scenario=event[:30],
        prediction=pred_metrics,
        keypoints=keypoint_metrics,
        explanation=exp_metrics,
        backtest=bt_metrics,
    )
    report.print_summary()

    # ── Phase 4: 事件级风险目标（独立于方向准确率）────────
    from evaluation.risk_metrics import RiskHead
    _risk = RiskHead(alpha=0.05).compute(
        point_path=list(pred_ret_real),
        lower_95=list(pred_lo95_real),
        upper_95=list(pred_hi95_real),
        dispersion=transmission.dispersion(),
        k=5,
    )
    _rk_msg = (f"  [风险] event_VaR={_risk.event_VaR:+.4f} "
               f"E[MDD]={_risk.expected_MDD:.4f} tail_prob={_risk.tail_prob:.3f} "
               f"区间=[{_risk.risk_interval[0]:+.4f},{_risk.risk_interval[1]:+.4f}]")
    logger.info(_rk_msg); print(_rk_msg)

    return {
        "chain":       chain.raw,
        "exog_factors": exog_factors,
        "transmission": transmission.to_dict(),   # Phase 1 张量视图（供 Phase 3/4 消费）
        "risk": _risk.to_dict(),                   # Phase 4 事件级风险目标（独立评测）
        "prediction": {
            "point_forecast":      pred.point_forecast.tolist(),
            "point_forecast_real": pred_ret_real.tolist(),
            "lower_95":            pred_lo95_real.tolist(),
            "upper_95":            pred_hi95_real.tolist(),
            "gate_weight":         pred.gate_weight,
        },
        "metrics":              json.loads(report.to_json()),
        "keypoints":            keypoint_metrics.summary,   # 扁平化，方便批量脚本消费
        "triggered_correction": len(triggered_list) > 0,
        "ablation_mode":        ABLATION_MODE,
        # ── Disagreement-as-feature support (added for paper §5.X) ──
        "disagreement": {
            "ts_pre_signal":           ts_pre_signal,    # "+" or "-" or ""
            "ts_pre_pred5_mean":       float(ts_pre_pred5),
            "llm_signal":              llm_signal,       # "+" or "-" or ""
            "net_signal_inject_value": float(net_signal_inject),
            "disagreement":            disagreement_flag,
            # variant-B debate: abstained? + judge confidence (for coverage curve)
            "debate_abstained":        bool(locals().get("_debate_info", {}).get("abstained", False)) if USE_DEBATE else None,
            "debate_confidence":       (locals().get("_debate_info", {}) or {}).get("confidence") if USE_DEBATE else None,
            # reldecomp intervention: the 3-step R_ind / R_mkt / beats-market
            "reld_r_ind":              (locals().get("_reld_info", {}) or {}).get("r_ind") if USE_RELDECOMP else None,
            "reld_r_mkt":              (locals().get("_reld_info", {}) or {}).get("r_mkt") if USE_RELDECOMP else None,
            "reld_beats_market":       (locals().get("_reld_info", {}) or {}).get("beats_market") if USE_RELDECOMP else None,
        },
        # ── Position-sizing experiment Sharpe variants (Plan Sharpe 修复) ──
        "backtest_v2_magnitude": {
            "SharpeRatio":      bt_metrics_v2.SharpeRatio,
            "AnnualizedReturn": bt_metrics_v2.AnnualizedReturn,
            "MaxDrawdown":      bt_metrics_v2.MaxDrawdown,
            "WinRate":          bt_metrics_v2.WinRate,
        },
        "backtest_v3_magnitude_stoploss": {
            "SharpeRatio":      bt_metrics_v3.SharpeRatio,
            "AnnualizedReturn": bt_metrics_v3.AnnualizedReturn,
            "MaxDrawdown":      bt_metrics_v3.MaxDrawdown,
            "WinRate":          bt_metrics_v3.WinRate,
        },
        # ── Cost-sensitivity Sharpe (Plan C, Stage 1.3) ─────────────
        "backtest_cost_010": {"SharpeRatio": bt_cost_010.SharpeRatio,
                              "WinRate": bt_cost_010.WinRate},
        "backtest_cost_020": {"SharpeRatio": bt_cost_020.SharpeRatio,
                              "WinRate": bt_cost_020.WinRate},
        "backtest_cost_030": {"SharpeRatio": bt_cost_030.SharpeRatio,
                              "WinRate": bt_cost_030.WinRate},
    }


# ──────────────────────────────────────────────
# CLI 入口
# ──────────────────────────────────────────────
if __name__ == "__main__":
    API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

    result = run_analysis(
        event="央行、金融监管总局、证监会联合发布一揽子金融政策，包括降准、降存量房贷利率、互换便利等重磅举措",
        entity_names=["央行一揽子政策", "计算机", "非银金融", "801750"],
        task_type="policy_event",
        industry_code="801750",
        event_date="2024-09-24",
        api_key=API_KEY,
    )

    print("\n[最终输出]")
    print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))

    car_df = pd.read_csv("data/processed/car_results.csv")
    real_car = car_df[
        (car_df["event_date"] == "2024-09-24") &
        (car_df["industry_code"].astype(str) == "801750") &
        (car_df["window"] == 5)
    ]
    if not real_car.empty:
        print(f"\n[对比] 真实 t+5 CAR = {real_car['CAR'].values[0]:.4f} "
              f"({real_car['CAR'].values[0] * 100:.2f}%)")
        point_forecast_real = result["prediction"]["point_forecast_real"]
        pred_mean_ret  = float(np.mean(point_forecast_real[:5]))
        pred_direction = "↑" if pred_mean_ret > 0 else "↓"
        real_direction = "↑" if real_car['CAR'].values[0] > 0 else "↓"
        print(f"[对比] 预测均值收益率={pred_mean_ret:.4f}  "
              f"预测方向={pred_direction}  真实方向={real_direction}  "
              f"{'✅ 方向正确' if pred_direction == real_direction else '❌ 方向错误'}")

    print("\n[最终输出]")
    print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))
