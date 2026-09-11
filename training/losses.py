"""
TopoLM: Modular Loss Functions
File: training/losses.py
"""

from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class DomainAssignmentLoss(nn.Module):
    """
    对比拓扑域归属损失 (Contrastive Domain Assignment Loss)。
    将各 Token 到底层所有候选域的负半马氏距离作为 Logits，施加交叉熵，在域之间构建 Voronoi 决策势垒。
    """
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(
        self,
        all_distances: torch.Tensor,
        target_domain_idx: int,
    ) -> torch.Tensor:
        """
        Args:
            all_distances: [num_domains, batch_size, seq_len] 到所有域的马氏距离平方
            target_domain_idx: 当前序列所属的目标域类别索引
        """
        num_domains = all_distances.size(0)
        # 转换形状至 [B * S, num_domains]，并以 -0.5 * d_M^2 作为未归一化对数概率
        logits_domain = -0.5 * all_distances.permute(1, 2, 0).reshape(-1, num_domains)
        labels = torch.full(
            (logits_domain.size(0),),
            target_domain_idx,
            dtype=torch.long,
            device=logits_domain.device,
        )
        return self.ce(logits_domain, labels)


class GaussianNLLLoss(nn.Module):
    """
    高斯负对数似然损失 (Gaussian Negative Log-Likelihood)。
    包含体积项 (log det Sigma) 与马氏发散项，强制模型在紧致的高斯流形内收敛，闭合体积漏洞。
    """
    def __init__(self):
        super().__init__()

    def forward(
        self,
        distance: torch.Tensor,
        log_sigma: torch.Tensor,
        domain_idx: int,
    ) -> torch.Tensor:
        """
        Args:
            distance: [batch_size, seq_len] 当前序列各 Token 到本域的马氏距离平方
            log_sigma: [num_domains, d_model] 域对角协方差的对数方差矩阵
            domain_idx: 当前活跃域索引
        """
        log_det = log_sigma[domain_idx].sum()  # 空间超体积正则项
        maha = distance.mean()
        return 0.5 * (maha + log_det)


class CentroidRepulsionLoss(nn.Module):
    """
    中心互斥损失 (Centroid Hinge Repulsion Loss)。
    保证互斥的公理/领域中心之间维持绝对几何安全裕度 (Margin)。
    """
    def __init__(self, margin: float = 8.0):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        centroids: torch.Tensor,
        disjoint_pairs: List[Tuple[int, int]],
    ) -> torch.Tensor:
        """
        Args:
            centroids: [num_domains, d_model] 概念中心点集合
            disjoint_pairs: 互斥领域索引对集合，如 [(DOMAIN_PHYSICS, DOMAIN_FANTASY)]
        """
        loss = torch.zeros((), device=centroids.device)
        if not disjoint_pairs:
            return loss

        for d1, d2 in disjoint_pairs:
            dist = torch.norm(centroids[d1] - centroids[d2], p=2)
            loss = loss + F.relu(self.margin - dist)

        return loss


class HierarchicalContainmentLoss(nn.Module):
    """
    层级包含软损失 (Hierarchical Containment Loss)。
    用于保证子域在几何尺度上完全内包于父域中 (如 Gravity ⊂ Physics ⊂ Reality)。
    """
    def __init__(self, tau: float = 1.0):
        super().__init__()
        self.tau = tau

    def forward(
        self,
        centroids: torch.Tensor,
        sigmas: torch.Tensor,
        inclusion_pairs: List[Tuple[int, int]],
    ) -> torch.Tensor:
        """
        Args:
            centroids: [num_domains, d_model]
            sigmas: [num_domains, d_model]
            inclusion_pairs: [(sub_domain_idx, super_domain_idx)]
        """
        loss = torch.zeros((), device=centroids.device)
        if not inclusion_pairs:
            return loss

        for sub_idx, super_idx in inclusion_pairs:
            u_sub, u_super = centroids[sub_idx], centroids[super_idx]
            r_sub = torch.sqrt(sigmas[sub_idx].sum())
            r_super = torch.sqrt(sigmas[super_idx].sum())
            dist = torch.norm(u_sub - u_super, p=2)

            # 约束目标: dist + r_sub <= r_super (外包络完全覆盖)
            violation = F.relu(dist + r_sub - r_super)
            loss = loss + violation

        return loss


class TopoLMLoss(nn.Module):
    """
    TopoLM 统一多目标联合损失调度器。
    L = L_LM + alpha_assign * L_assign + alpha_nll * L_nll + beta_rel * L_rel + beta_inc * L_inc
    """
    def __init__(
        self,
        alpha_assign: float = 1.0,
        alpha_nll: float = 0.05,
        beta_rel: float = 0.5,
        beta_inc: float = 0.5,
        margin: float = 8.0,
    ):
        super().__init__()
        self.alpha_assign = alpha_assign
        self.alpha_nll = alpha_nll
        self.beta_rel = beta_rel
        self.beta_inc = beta_inc

        self.ce_loss = nn.CrossEntropyLoss()
        self.assign_loss_fn = DomainAssignmentLoss()
        self.nll_loss_fn = GaussianNLLLoss()
        self.rel_loss_fn = CentroidRepulsionLoss(margin=margin)
        self.inc_loss_fn = HierarchicalContainmentLoss()

    def forward(
        self,
        domain_outputs: Dict[int, Dict[str, torch.Tensor]],
        targets_by_domain: Dict[int, torch.Tensor],
        topo_space: nn.Module,
        disjoint_pairs: List[Tuple[int, int]] = [],
        inclusion_pairs: List[Tuple[int, int]] = [],
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            domain_outputs: 各域前向传播的输出集合字典 {domain_idx: output_dict}
            targets_by_domain: 各域自回归目标 {domain_idx: targets}
            topo_space: 拓扑高斯空间实例 (包含 mu, log_sigma)
            disjoint_pairs: 互斥域列表
            inclusion_pairs: 包含域列表
        """
        device = topo_space.mu.device
        lm_loss = torch.zeros((), device=device)
        assign_loss = torch.zeros((), device=device)
        nll_loss = torch.zeros((), device=device)

        num_active_domains = len(domain_outputs)
        for domain_idx, output in domain_outputs.items():
            # 1. 语言自回归交叉熵
            logits = output["logits"]
            targets = targets_by_domain[domain_idx]
            lm_loss = lm_loss + self.ce_loss(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )

            # 2. 对比域分配损失
            assign_loss = assign_loss + self.assign_loss_fn(
                output["all_distances"],
                domain_idx,
            )

            # 3. 高斯体积约束负对数似然
            nll_loss = nll_loss + self.nll_loss_fn(
                output["distance"],
                topo_space.log_sigma,
                domain_idx,
            )

        lm_loss = lm_loss / num_active_domains
        assign_loss = assign_loss / num_active_domains
        nll_loss = nll_loss / num_active_domains

        # 4. 跨域拓扑关系约束 (排斥 + 包含)
        rel_loss = self.rel_loss_fn(topo_space.mu, disjoint_pairs)
        inc_loss = self.inc_loss_fn(topo_space.mu, topo_space.get_sigma(), inclusion_pairs)

        total_loss = (
            lm_loss
            + self.alpha_assign * assign_loss
            + self.alpha_nll * nll_loss
            + self.beta_rel * rel_loss
            + self.beta_inc * inc_loss
        )

        return {
            "total": total_loss,
            "lm": lm_loss,
            "assign": assign_loss,
            "nll": nll_loss,
            "relation": rel_loss,
            "inclusion": inc_loss,
        }