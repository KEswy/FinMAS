"""
KF-Transformer — 状态空间 × Transformer 滤波分支
+ GatedFusion 门控融合（与 Informer 双路协同）
"""
from __future__ import annotations
import torch
import torch.nn as nn
from config import CONFIG, KFTransformerConfig


# ──────────────────────────────────────────────
# KF-Transformer
# ──────────────────────────────────────────────
class KFTransformer(nn.Module):
    """
    Transformer Encoder 估计卡尔曼隐状态与过程噪声参数，
    输出预测均值与方差（用于区间校准）。
    """

    def __init__(self, cfg: KFTransformerConfig = None):
        super().__init__()
        cfg = cfg or CONFIG.kf_transformer
        self.cfg = cfg
        self.pred_len = cfg.pred_len

        # 输入投影
        self.input_proj = nn.Linear(cfg.obs_dim, cfg.d_model)

        # Transformer Encoder（估计隐状态）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model, nhead=cfg.n_heads,
            dim_feedforward=cfg.d_model * 4, dropout=cfg.dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.n_layers)

        # 隐状态估计头
        self.state_head = nn.Linear(cfg.d_model, cfg.state_dim)

        # KF 参数估计头（过程噪声 Q 的对角元素）
        self.process_noise_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.state_dim),
            nn.Softplus()   # 保证正定
        )

        # 观测噪声 R（标量，可学习）
        self.log_obs_noise = nn.Parameter(torch.zeros(1))

        # 预测解码器
        self.decoder = nn.Sequential(
            nn.Linear(cfg.state_dim, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.pred_len * 2)  # 均值 + 方差
        )

    def forward(self, x: torch.Tensor):
        """
        x: (B, seq_len, obs_dim)
        returns:
            mean: (B, pred_len, 1)
            var:  (B, pred_len, 1)   — 预测方差
        """
        B, T, _ = x.shape
        h = self.input_proj(x)           # (B, T, d_model)
        h = self.encoder(h)              # (B, T, d_model)
        h_last = h[:, -1, :]             # 取最后时刻

        state = self.state_head(h_last)  # (B, state_dim)
        Q_diag = self.process_noise_head(h_last)  # 过程噪声

        out = self.decoder(state)        # (B, pred_len*2)
        out = out.view(B, self.pred_len, 2)

        mean = out[:, :, :1]
        log_var = out[:, :, 1:]
        # 观测噪声叠加
        obs_noise = self.log_obs_noise.exp()
        var = log_var.exp() + obs_noise

        return mean, var

    @torch.no_grad()
    def predict(self, x: torch.Tensor):
        self.eval()
        return self.forward(x)

    def confidence_interval(self, mean: torch.Tensor, var: torch.Tensor,
                            alpha: float = 0.95):
        """返回 (lower, upper) 置信区间"""
        import math
        z = {0.80: 1.282, 0.90: 1.645, 0.95: 1.960, 0.99: 2.576}.get(alpha, 1.960)
        std = var.sqrt()
        return mean - z * std, mean + z * std


# ──────────────────────────────────────────────
# 门控融合
# ──────────────────────────────────────────────
class GatedFusion(nn.Module):
    """
    自适应融合 Informer（点预测）与 KF-Transformer（均值+方差）
    门控输入：事件强度、市场波动、模型残差
    """

    def __init__(self, pred_len: int, signal_dim: int = 3):
        """
        signal_dim: 门控信号维度
            [event_strength, market_volatility, residual_magnitude]
        """
        super().__init__()
        self.pred_len = pred_len
        self.gate_net = nn.Sequential(
            nn.Linear(signal_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        informer_pred: torch.Tensor,   # (B, pred_len, 1)
        kf_mean: torch.Tensor,          # (B, pred_len, 1)
        kf_var: torch.Tensor,           # (B, pred_len, 1)
        gate_signal: torch.Tensor,      # (B, signal_dim)
    ):
        """
        返回:
            fused_mean:  (B, pred_len, 1)
            fused_var:   (B, pred_len, 1)
            gate_weight: (B, 1, 1)  — KF 分支权重
        """
        w = self.gate_net(gate_signal).unsqueeze(1)  # (B, 1, 1)
        # w → KF 分支权重；(1-w) → Informer 分支权重
        fused_mean = w * kf_mean + (1 - w) * informer_pred
        # 融合方差（混合高斯近似）
        fused_var = (
            w * (kf_var + (kf_mean - fused_mean) ** 2) +
            (1 - w) * (kf_mean - fused_mean) ** 2   # Informer 无方差，用偏差近似
        )
        return fused_mean, fused_var, w

    @staticmethod
    def make_gate_signal(
        event_strength: float,
        market_volatility: float,
        residual_magnitude: float,
        device: str = "cpu"
    ) -> torch.Tensor:
        """构造门控信号张量"""
        return torch.tensor(
            [[event_strength, market_volatility, residual_magnitude]],
            dtype=torch.float32, device=device
        )
