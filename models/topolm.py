"""
TopoLM: Unified Modular Architecture
File: models/topolm.py
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
from models.domain_space import TopologicalDomainSpace
from models.domain_router import DomainRouter
from models.spatial_attention import DenseTopologicalAttention, SparseIndexedAttention
from models.constraint_layer import RelativeVoronoiConstraintLayer


class TopoLM(nn.Module):
    """
    TopoLM 端到端模型：
    结合词嵌入、绝对位置编码、多元高斯拓扑域空间、空间注意力机制与前馈解码层。
    """
    def __init__(
        self,
        vocab_size: int,
        num_domains: int = 2,
        d_model: int = 64,
        num_heads: int = 4,
        max_seq_len: int = 128,
        init_log_sigma: float = 0.5,
        topology_temperature: float = 1.0,
        topology_bias_scale: float = 1.0,
        sparse_mode: bool = False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.sparse_mode = sparse_mode

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        self.topo_space = TopologicalDomainSpace(num_domains, d_model, init_log_sigma)
        self.router = DomainRouter()
        
        if sparse_mode:
            self.spatial_attn = SparseIndexedAttention(d_model, num_heads)
        else:
            self.spatial_attn = DenseTopologicalAttention(
                d_model, num_heads, topology_temperature, topology_bias_scale
            )

        self.ln1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        self.constraint_layer = RelativeVoronoiConstraintLayer()

    def forward(
        self,
        input_ids: torch.Tensor,
        active_domain_idx: int,
        apply_constraint: bool = False,
    ) -> Dict[str, torch.Tensor]:
        batch_size, seq_len = input_ids.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length ({seq_len}) exceeds max_seq_len ({self.max_seq_len})")

        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)

        # 分离纯语义嵌入与融入位置编码的注意力输入
        semantic_x = self.token_emb(input_ids)
        attention_x = semantic_x + self.pos_emb(positions)

        # 拓扑度量计算
        distance = self.topo_space.mahalanobis_sq(semantic_x, active_domain_idx)
        all_distances = self.topo_space.compute_all_distances(semantic_x)

        if self.sparse_mode:
            active_indices, probs = self.router.route_active_tokens(
                semantic_x, self.topo_space, active_domain_idx
            )
            attn_out, attn_weights = self.spatial_attn(attention_x, active_indices)
        else:
            probs = torch.exp(-0.5 * distance)
            attn_out, attn_weights = self.spatial_attn(attention_x, probs)

        hidden = self.ln1(attention_x + attn_out)
        hidden = self.ln2(hidden + self.ffn(hidden))
        logits = self.lm_head(hidden)

        if apply_constraint:
            logits[:, -1, :] = self.constraint_layer(
                logits[:, -1, :], self.token_emb.weight, self.topo_space, active_domain_idx
            )

        return {
            "logits": logits,
            "membership": probs,
            "distance": distance,
            "all_distances": all_distances,
            "semantic_x": semantic_x,
            "attention": attn_weights,
        }