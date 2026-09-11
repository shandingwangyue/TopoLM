"""
TopoLM: Domain Inverted Index Router
File: models/domain_router.py
"""

from typing import Tuple
import torch
import torch.nn as nn


class DomainRouter(nn.Module):
    """
    负责将上下文序列投影到拓扑域空间，动态提取满足高斯包含条件的激活 Token 索引 (Active Keys)。
    """
    def __init__(self, threshold: float = 1e-4, min_active_k: int = 4):
        super().__init__()
        self.threshold = threshold
        self.min_active_k = min_active_k

    @torch.no_grad()
    def route_active_tokens(
        self,
        semantic_x: torch.Tensor,
        topo_space: nn.Module,
        active_domain_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            semantic_x: [batch_size, seq_len, d_model] 词嵌入语义向量
            topo_space: 拓扑空间实例
            active_domain_idx: 当前激活的主域索引
        Returns:
            active_indices: [batch_size, K_active] 升序排序的激活键索引
            inclusion_probs: [batch_size, seq_len] 隶属度概率
        """
        batch_size, seq_len, _ = semantic_x.shape
        inclusion_probs = topo_space.membership_prob(semantic_x, active_domain_idx)

        # 选出大于阈值的 Token，如果过少则保底选取 Top-K
        mask = inclusion_probs >= self.threshold
        active_list = []

        for b in range(batch_size):
            valid_idx = torch.nonzero(mask[b], as_tuple=False).squeeze(-1)
            if valid_idx.numel() < self.min_active_k:
                # 兜底：若全被拦截，按距离最近选取前 min_active_k 个候选
                _, topk_idx = torch.topk(inclusion_probs[b], k=min(self.min_active_k, seq_len))
                valid_idx = topk_idx
            valid_idx, _ = torch.sort(valid_idx)
            active_list.append(valid_idx)

        # 对齐 Batch 维度，填充至当前批次最大激活长度
        max_k = max(len(idx) for idx in active_list)
        padded_indices = torch.zeros((batch_size, max_k), dtype=torch.long, device=semantic_x.device)
        for b in range(batch_size):
            cur_len = len(active_list[b])
            padded_indices[b, :cur_len] = active_list[b]
            if cur_len < max_k:
                # 尾部用最后一个有效索引填充 (因果对齐安全)
                padded_indices[b, cur_len:] = active_list[b][-1]

        return padded_indices, inclusion_probs