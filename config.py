"""
全局配置 — 多智能体金融分析系统
"""
import os
from dataclasses import dataclass, field
from typing import List

# ──────────────────────────────────────────────
# LLM
# ──────────────────────────────────────────────
@dataclass
class LLMConfig:
    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 2048
    temperature: float = 0.2          # 金融推理要低温
    api_base: str = "https://api.anthropic.com/v1/messages"

# ──────────────────────────────────────────────
# RAG / 检索
# ──────────────────────────────────────────────
@dataclass
class RAGConfig:
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k_dense: int = 8
    top_k_sparse: int = 8
    top_k_graph: int = 5
    rerank_top_k: int = 5
    # 向量库：faiss / milvus / chroma
    vector_store: str = "faiss"
    embedding_model: str = "BAAI/bge-large-zh-v1.5"
    # 图数据库
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"

# ──────────────────────────────────────────────
# 知识图谱
# ──────────────────────────────────────────────
@dataclass
class KGConfig:
    # 传导层级: 社会 → 行业 → 公司 → 资产
    layers: List[str] = field(
        default_factory=lambda: ["social", "industry", "company", "asset"]
    )
    # 边权衰减半衰期（天）
    decay_half_life: int = 30
    # 传导权重阈值（低于此值截断）
    edge_weight_threshold: float = 0.05
    # hop 数
    max_hops: int = 3

# ──────────────────────────────────────────────
# 时序模型 (Informer)
# ──────────────────────────────────────────────
@dataclass
class InformerConfig:
    seq_len: int = 96          # 输入窗口
    label_len: int = 48        # Decoder start token
    pred_len: int = 24         # 预测步长
    # 6 个市场特征 + 2 个外生槽位（event_strength / net_signal）
    enc_in: int = 8
    dec_in: int = 8
    c_out: int = 1
    d_model: int = 512
    n_heads: int = 8
    e_layers: int = 2
    d_layers: int = 1
    d_ff: int = 2048
    factor: int = 5            # ProbSparse 采样因子
    dropout: float = 0.05
    attn: str = "prob"         # prob | full
    activation: str = "gelu"
    distil: bool = True        # 蒸馏层
    # ── 双头：回归 + 分类 ───────────────────────
    # 在回归头之外额外训练一个二分类头（涨/跌），仅作为 MTL 正则化
    # 推理时仍以回归头为主（保持预测管线不变）
    cls_weight: float = 0.3    # BCE 损失权重 (loss = MSE + cls_weight * BCE)
    cls_deadzone: float = 0.1  # |y_standardized| < 此阈值的样本不计 BCE（忽略噪声样本）

# ──────────────────────────────────────────────
# KF-Transformer
# ──────────────────────────────────────────────
@dataclass
class KFTransformerConfig:
    state_dim: int = 32        # 隐状态维度
    obs_dim: int = 8           # 观测维度（与 Informer 输入维度一致）
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 2
    pred_len: int = 24
    dropout: float = 0.1

# ──────────────────────────────────────────────
# 评测
# ──────────────────────────────────────────────
@dataclass
class EvalConfig:
    # 预测指标
    metrics: List[str] = field(
        default_factory=lambda: ["MAE", "MSE", "RMSE", "MAPE", "DirectionAcc"]
    )
    # 区间校准
    confidence_levels: List[float] = field(
        default_factory=lambda: [0.8, 0.9, 0.95]
    )
    # 链条质量
    chain_metrics: List[str] = field(
        default_factory=lambda: ["ConsistencyScore", "EvidenceCoverage", "GraphConsistency"]
    )
    # 回测
    backtest_cost: float = 0.001     # 单边交易成本
    backtest_slippage: float = 0.0005

# ──────────────────────────────────────────────
# 主配置
# ──────────────────────────────────────────────
@dataclass
class SystemConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    kg: KGConfig = field(default_factory=KGConfig)
    informer: InformerConfig = field(default_factory=InformerConfig)
    kf_transformer: KFTransformerConfig = field(default_factory=KFTransformerConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    # 触发反向校准的残差阈值（可被 RESIDUAL_TRIGGER_THRESHOLD 环境变量覆盖，用于 B2 闭环关闭消融）
    residual_trigger_threshold: float = field(
        default_factory=lambda: float(os.environ.get("RESIDUAL_TRIGGER_THRESHOLD", "0.05"))
    )
    # 工作目录
    workspace: str = "./workspace"
    log_level: str = "INFO"

CONFIG = SystemConfig()
