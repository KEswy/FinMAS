"""
Informer — ProbSparse Self-Attention + Distilling + Generative Decoder
参考: Zhou et al., AAAI 2021
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import CONFIG, InformerConfig


# ──────────────────────────────────────────────
# ProbSparse Self-Attention
# ──────────────────────────────────────────────
class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, dropout=0.1, output_attention=False):
        super().__init__()
        self.factor = factor
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):
        """从 Q 中选取 Top-u 个稀疏查询"""
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # 随机采样 K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k))
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        # 稀疏性度量：max - mean
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # 用 Top-u Q 计算注意力
        Q_reduce = Q[
            torch.arange(B)[:, None, None],
            torch.arange(H)[None, :, None],
            M_top, :
        ]
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))
        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        """用均值初始化未被选中位置的上下文"""
        B, H, L_V, D = V.shape
        if self.mask_flag:
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape
        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -1e9)

        attn = torch.softmax(scores, dim=-1)
        context_in[
            torch.arange(B)[:, None, None],
            torch.arange(H)[None, :, None],
            index, :
        ] = torch.matmul(attn, V).type_as(context_in)

        return context_in, attn if self.output_attention else None

    def forward(self, queries, keys, values, attn_mask=None):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * math.ceil(math.log(L_K))
        u = self.factor * math.ceil(math.log(L_Q))
        U_part = min(U_part, L_K)
        u = min(u, L_Q)

        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)
        scale = 1.0 / math.sqrt(D)
        scores_top = scores_top * scale

        context = self._get_initial_context(values, L_Q)
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)

        return context.contiguous(), attn


class ProbMask:
    def __init__(self, B, H, L, index, scores, device):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[
            torch.arange(B)[:, None, None],
            torch.arange(H)[None, :, None],
            index, :
        ]
        self.mask = indicator.view(scores.shape)

# ──────────────────────────────────────────────
# Full Attention（用于 Decoder cross-attention）
# ──────────────────────────────────────────────
class FullAttention(nn.Module):
    def __init__(self, mask_flag=False, dropout=0.1, output_attention=False):
        super().__init__()
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, attn_mask=None):
        # 输入格式与 ProbAttention 一致：(B, L, H, D)
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        # 转为 (B, H, L, D) 再做 attention
        queries = queries.transpose(2, 1)   # (B, H, L_Q, D)
        keys    = keys.transpose(2, 1)      # (B, H, L_K, D)
        values  = values.transpose(2, 1)    # (B, H, L_K, D)

        scale = 1.0 / (D ** 0.5)
        scores = torch.matmul(queries, keys.transpose(-2, -1)) * scale  # (B, H, L_Q, L_K)

        attn = self.dropout(torch.softmax(scores, dim=-1))
        out  = torch.matmul(attn, values)   # (B, H, L_Q, D)

        # 转回 (B, L_Q, H*D) 供 AttentionLayer 的 out_projection 使用
        out = out.contiguous().view(B, L_Q, -1)  # (B, L_Q, H*D)
        # AttentionLayer 要求返回 (context, attn)，context shape = (B, L, H, D)
        out = out.view(B, L_Q, H, D)

        return out, attn if self.output_attention else None
# ──────────────────────────────────────────────
# Attention Layer
# ──────────────────────────────────────────────
class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super().__init__()
        d_keys = d_keys or d_model // n_heads
        d_values = d_values or d_model // n_heads
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask=None):
        B, L, _ = queries.shape
        S = keys.shape[1]
        H = self.n_heads
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        out, attn = self.inner_attention(queries, keys, values, attn_mask)
        out = out.view(B, L, -1)
        return self.out_projection(out), attn


# ──────────────────────────────────────────────
# Encoder 蒸馏层
# ──────────────────────────────────────────────
class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super().__init__()
        self.downConv = nn.Conv1d(c_in, c_in, 3, padding=1, padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        return x.transpose(1, 2)


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation='relu'):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(d_model, d_ff, 1)
        self.conv2 = nn.Conv1d(d_ff, d_model, 1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == 'relu' else F.gelu

    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(x, x, x, attn_mask)
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y), attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super().__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        attns = []
        if self.conv_layers:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, attn_mask)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask)
                attns.append(attn)
        if self.norm:
            x = self.norm(x)
        return x, attns


# ──────────────────────────────────────────────
# Decoder
# ──────────────────────────────────────────────
class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation='relu'):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(d_model, d_ff, 1)
        self.conv2 = nn.Conv1d(d_ff, d_model, 1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == 'relu' else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x = x + self.dropout(self.self_attention(x, x, x, x_mask)[0])
        x = self.norm1(x)
        x = x + self.dropout(self.cross_attention(x, cross, cross, cross_mask)[0])
        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm3(x + y)


class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask, cross_mask)
        if self.norm:
            x = self.norm(x)
        if self.projection:
            x = self.projection(x)
        return x


# ──────────────────────────────────────────────
# 位置编码
# ──────────────────────────────────────────────
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super().__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_mark=None):
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)


# ──────────────────────────────────────────────
# Informer 主模型
# ──────────────────────────────────────────────
class Informer(nn.Module):
    def __init__(self, cfg: InformerConfig = None):
        super().__init__()
        cfg = cfg or CONFIG.informer
        self.pred_len = cfg.pred_len
        self.label_len = cfg.label_len

        # Embedding
        self.enc_embedding = DataEmbedding(cfg.enc_in, cfg.d_model, cfg.dropout)
        self.dec_embedding = DataEmbedding(cfg.dec_in, cfg.d_model, cfg.dropout)

        # Encoder
        Attn = ProbAttention
        enc_attn_layers = [
            EncoderLayer(
                AttentionLayer(
                    Attn(True, cfg.factor, cfg.dropout),
                    cfg.d_model, cfg.n_heads
                ),
                cfg.d_model, cfg.d_ff, cfg.dropout, cfg.activation
            ) for _ in range(cfg.e_layers)
        ]
        enc_conv_layers = (
            [ConvLayer(cfg.d_model) for _ in range(cfg.e_layers - 1)]
            if cfg.distil else None
        )
        self.encoder = Encoder(
            enc_attn_layers, enc_conv_layers,
            norm_layer=nn.LayerNorm(cfg.d_model)
        )

        # Decoder — projection=None，投影改到外部双头处理
        dec_attn_layers = [
            DecoderLayer(
                AttentionLayer(ProbAttention(True, cfg.factor, cfg.dropout), cfg.d_model, cfg.n_heads),
                AttentionLayer(FullAttention(False, cfg.dropout), cfg.d_model, cfg.n_heads),
                cfg.d_model, cfg.d_ff, cfg.dropout, cfg.activation
            ) for _ in range(cfg.d_layers)
        ]
        self.decoder = Decoder(
            dec_attn_layers,
            norm_layer=nn.LayerNorm(cfg.d_model),
            projection=None,   # 双头：外部分别投影
        )

        # 双头：回归（收益率点预测） + 分类（涨/跌 logit）
        # 保留 c_out=1 的回归头不变；cls 头为二分类 logit（用 BCEWithLogits）
        self.reg_head = nn.Linear(cfg.d_model, cfg.c_out)
        self.cls_head = nn.Linear(cfg.d_model, 1)

    def _shared_trunk(self, x_enc: torch.Tensor, x_dec: torch.Tensor) -> torch.Tensor:
        """Encoder + Decoder，返回共享主干表示 (B, pred_len, d_model)"""
        enc_out = self.enc_embedding(x_enc)
        enc_out, _ = self.encoder(enc_out)

        dec_out = self.dec_embedding(x_dec)
        dec_out = self.decoder(dec_out, enc_out)   # (B, label_len+pred_len, d_model)
        return dec_out[:, -self.pred_len:, :]       # (B, pred_len, d_model)

    def forward(self, x_enc, x_dec):
        """
        x_enc: (B, seq_len, enc_in)
        x_dec: (B, label_len+pred_len, dec_in)
        returns: (B, pred_len, c_out)  — 仅回归输出（保持向后兼容）
        """
        trunk = self._shared_trunk(x_enc, x_dec)
        return self.reg_head(trunk)

    def forward_dual(self, x_enc, x_dec):
        """
        双头前向：同时返回回归输出与分类 logit。
        - 训练时使用，用于 MTL (MSE + BCE) 联合优化
        - 推理时仍建议使用 `forward` / `predict`，只取回归头

        returns:
            reg_out:   (B, pred_len, c_out)
            cls_logit: (B, pred_len, 1)
        """
        trunk = self._shared_trunk(x_enc, x_dec)
        reg_out   = self.reg_head(trunk)
        cls_logit = self.cls_head(trunk)
        return reg_out, cls_logit

    @torch.no_grad()
    def predict(self, x_enc: torch.Tensor, x_dec: torch.Tensor) -> torch.Tensor:
        self.eval()
        return self.forward(x_enc, x_dec)

    @torch.no_grad()
    def predict_dual(self, x_enc: torch.Tensor, x_dec: torch.Tensor):
        """返回 (reg_out, cls_prob)，分类概率经过 sigmoid"""
        self.eval()
        reg_out, cls_logit = self.forward_dual(x_enc, x_dec)
        return reg_out, torch.sigmoid(cls_logit)
