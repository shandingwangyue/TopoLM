"""
TopoLM: Spatial Attention Mechanisms
File: models/spatial_attention.py
"""

from typing import Dict, Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseTopologicalAttention(nn.Module):
    """
    Phase-1 / Phase-2: 带有软拓扑偏置的稠密因果注意力算子 (O(N^2))。
    在注意力打分矩阵上直接注入属于当前拓扑域的对数先验偏置。
    """
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        topology_temperature: float = 1.0,
        topology_bias_scale: float = 1.0,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.topology_temperature = topology_temperature
        self.topology_bias_scale = topology_bias_scale

        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        attention_x: torch.Tensor,
        topo_inclusion_prob: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            attention_x: [batch_size, seq_len, d_model] 融入位置编码的表征
            topo_inclusion_prob: [batch_size, seq_len] 纯语义向量到当前域的高斯包含概率
        Returns:
            out: [batch_size, seq_len, d_model]
            attn_weights: [batch_size, num_heads, seq_len, seq_len]
        """
        batch_size, seq_len, _ = attention_x.size()

        qkv = self.qkv_proj(attention_x).chunk(3, dim=-1)
        q, k, v = [
            t.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            for t in qkv
        ]

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 注入拓扑软偏置: log(P(x in D)) / tau
        topo_bias = torch.log(topo_inclusion_prob.clamp_min(1e-8)) / self.topology_temperature
        topo_bias = (self.topology_bias_scale * topo_bias).view(batch_size, 1, 1, seq_len)
        scores = scores + topo_bias

        # 下三角自回归因果掩码
        causal_mask = torch.tril(
            torch.ones(seq_len, seq_len, device=attention_x.device, dtype=torch.bool)
        ).view(1, 1, seq_len, seq_len)

        scores = scores.masked_fill(~causal_mask, torch.finfo(scores.dtype).min)
        attn_weights = F.softmax(scores, dim=-1)

        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        return self.out_proj(out), attn_weights


class SparseIndexedAttention(nn.Module):
    """
    Phase-3: 真正的非对称拓扑稀疏矩形注意力算子 (O(N * K_active))。
    仅依据拓扑路由结果 Gather 激活的候选键值对，消除 N x N 计算与显存开销。
    """
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        x: torch.Tensor,
        active_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch_size, seq_len, d_model] 完整序列张量
            active_indices: [batch_size, k_active] 拓扑域路由动态抽取的激活 Token 索引 (升序)
        Returns:
            out: [batch_size, seq_len, d_model]
            attn_weights: [batch_size, num_heads, seq_len, k_active] 矩形注意力权重
        """
        batch_size, seq_len, _ = x.shape
        k_active = active_indices.shape[1]

        qkv = self.qkv_proj(x).chunk(3, dim=-1)

        # 1. 完整投影 Q: [B, H, N, D]
        q = qkv[0].view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 2. 局部聚集 K, V (在序列维度裁剪，避免全局展开)
        k_full = qkv[1].view(batch_size, seq_len, self.num_heads, self.head_dim)
        v_full = qkv[2].view(batch_size, seq_len, self.num_heads, self.head_dim)

        idx_expanded = active_indices.unsqueeze(-1).unsqueeze(-1).expand(
            batch_size, k_active, self.num_heads, self.head_dim
        )
        k_active_t = torch.gather(k_full, dim=1, index=idx_expanded).transpose(1, 2)  # [B, H, K, D]
        v_active_t = torch.gather(v_full, dim=1, index=idx_expanded).transpose(1, 2)  # [B, H, K, D]

        # 3. 核心计算：Q [N] x K_active^T [K] -> 矩形投影 [B, H, N, K]
        scores = torch.matmul(q, k_active_t.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 4. 矩形因果掩码 (Rectangular Causal Mask)
        # 条件：只有当激活键的位置 pos_k <= 查询位置 pos_q 时才允许交互
        pos_q = torch.arange(seq_len, device=x.device).unsqueeze(1)               # [N, 1]
        pos_k = active_indices                                                    # [B, K]
        causal_mask = pos_q >= pos_k.unsqueeze(1)                                 # [B, N, K]

        scores = scores.masked_fill(
            ~causal_mask.unsqueeze(1),
            torch.finfo(scores.dtype).min
        )

        attn_weights = F.softmax(scores, dim=-1)
        # 边界防溢出保护：处理完全被因果掩码阻隔的初始位置
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        # 5. 加权汇聚：[B, H, N, K] x [B, H, K, D] -> [B, H, N, D]
        out = torch.matmul(attn_weights, v_active_t)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        return self.out_proj(out), attn_weights