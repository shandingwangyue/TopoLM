"""
TopoLM: Unified Modular Architecture
File: models/topolm.py
"""

from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from .domain_space import TopologicalDomainSpace
from .domain_router import DomainRouter
from .spatial_attention import DenseTopologicalAttention, SparseIndexedAttention
from .constraint_layer import RelativeVoronoiConstraintLayer


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
        router_threshold: float = 1e-4,
        router_min_active_k: int = 4,
        attention_metric: str = "wasserstein",
        wasserstein_temperature: float = 0.5,
        token_geometry: bool = False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.sparse_mode = sparse_mode
        self.num_domains = num_domains

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        if token_geometry:
            self.token_log_radius = nn.Embedding(vocab_size, 1)
            nn.init.normal_(self.token_emb.weight, std=0.01)
            nn.init.constant_(self.token_log_radius.weight, -2.0)
        else:
            self.token_log_radius = None

        self.topo_space = TopologicalDomainSpace(num_domains, d_model, init_log_sigma)
        self.router = DomainRouter(router_threshold, router_min_active_k)
        
        if sparse_mode:
            self.spatial_attn = SparseIndexedAttention(
                d_model, num_heads, attention_metric, wasserstein_temperature
            )
        else:
            self.spatial_attn = DenseTopologicalAttention(
                d_model, num_heads, topology_temperature, topology_bias_scale,
                attention_metric, wasserstein_temperature
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

    def concept_radii(self, token_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Positive radii of the optional per-token concept spheres."""
        if self.token_log_radius is None:
             raise ValueError("Model was not initialized with token_geometry=True")
        log_radius = self.token_log_radius.weight if token_ids is None else self.token_log_radius(token_ids)
        return F.softplus(log_radius.squeeze(-1)) + 0.1

    def forward(
        self,
        input_ids: torch.Tensor,
        active_domain_idx: int,
        apply_constraint: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if input_ids.ndim != 2 or input_ids.numel() == 0:
            raise ValueError("input_ids must be a nonempty [batch, sequence] tensor")
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
            active_indices, probs, active_mask = self.router.route_active_tokens(
                semantic_x, self.topo_space, active_domain_idx, return_mask=True
            )
            attn_out, attn_weights = self.spatial_attn(attention_x, active_indices, active_mask)
        else:
            probs = torch.exp(-0.5 * distance)
            attn_out, attn_weights = self.spatial_attn(
                attention_x, probs, log_membership=-0.5 * distance
            )

        hidden = self.ln1(attention_x + attn_out)
        hidden = self.ln2(hidden + self.ffn(hidden))
        logits = self.lm_head(hidden)

        if apply_constraint:
            last_logits = self.constraint_layer(
                logits[:, -1, :], self.token_emb.weight, self.topo_space, active_domain_idx
            )
            logits = torch.cat([logits[:, :-1, :], last_logits.unsqueeze(1)], dim=1)

        return {
            "logits": logits,
            "membership": probs,
            "distance": distance,
            "all_distances": all_distances,
            "semantic_x": semantic_x,
            "attention": attn_weights,
        }
