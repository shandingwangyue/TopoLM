"""
TopoLM: Relative Voronoi Constraint Layer
File: models/constraint_layer.py
"""

import torch
import torch.nn as nn


class RelativeVoronoiConstraintLayer(nn.Module):
    """
    在模型自回归输出层实施相对拓扑门控。
    只要候选 Token 到本域的马氏距离大于到任何异域的距离，判定为拓扑排斥，强制剥夺输出概率。
    """
    def __init__(self, penalty_val: float = -1e4):
        super().__init__()
        self.penalty_val = penalty_val

    def forward(
        self,
        logits: torch.Tensor,
        vocab_embeddings: torch.Tensor,
        topo_space: nn.Module,
        active_domain_idx: int,
    ) -> torch.Tensor:
        """
        Args:
            logits: [batch_size, vocab_size] 输出层原始得分
            vocab_embeddings: [vocab_size, d_model] 静态词表嵌入矩阵
            topo_space: 拓扑空间实例
            active_domain_idx: 当前激活域索引
        Returns:
            constrained_logits: 经过 Voronoi 决策面过滤后的得分
        """
        # 计算整张词表到所有域的马氏距离 [num_domains, vocab_size]
        all_dists = topo_space.compute_all_distances(vocab_embeddings.unsqueeze(0)).squeeze(1)

        target_dist = all_dists[active_domain_idx]
        
        # 异域最小距离 (排除自身)
        mask = torch.ones(topo_space.num_domains, dtype=torch.bool, device=logits.device)
        mask[active_domain_idx] = False
        min_other_dist, _ = torch.min(all_dists[mask], dim=0)

        # 相对拓扑判据：到本域距离 >= 到异域最近距离，判定为非法异域词
        forbidden_mask = target_dist >= min_other_dist

        # 广播并掩盖非法 Token
        constrained_logits = logits.masked_fill(forbidden_mask.unsqueeze(0), self.penalty_val)
        return constrained_logits