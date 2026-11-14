"""
TopoLM: Phase 1 Physical Ontology Emergence Loss
File: training/phase1_loss.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class TopoMultiObjectiveLoss(nn.Module):
    """
    统一多目标拓扑损失函数 (Unified Multi-Objective Topological Loss)
    严格对应论文 Section 5 公式 (5) 与 Section 9.1 物理涌现势能:
    L = L_LM + alpha * L_mem (Surface Tension) + beta * L_rel (Inclusion + Repulsion) + gamma * L_con
    """
    def __init__(
        self,
        delta_inc: float = 0.05,    # 包含边界安全壳裕度
        delta_rep: float = 2.0,     # 互斥排斥最小边界距离
        alpha_tension: float = 0.01,# 表面张力/体积守恒权重 (防止体积膨胀漏洞 Volume Loophole)
        beta_rel: float = 1.0,      # 关系结构损失权重
        gamma_con: float = 0.1      # 宏观域一致性惩罚权重
    ):
        super().__init__()
        self.delta_inc = delta_inc
        self.delta_rep = delta_rep
        self.alpha_tension = alpha_tension
        self.beta_rel = beta_rel
        self.gamma_con = gamma_con

    def inclusion_loss(self, mu_sub, r_sub, mu_super, r_super) -> torch.Tensor:
        """
        非对称包含势能 (Asymmetric Inclusion Energy for A \subset B):
        max(0, ||mu_A - mu_B||_2 + R_A + delta_inc - R_B)
        """
        dist = torch.norm(mu_sub - mu_super, p=2, dim=-1)
        violation = F.relu(dist + r_sub + self.delta_inc - r_super)
        return violation.mean()

    def repulsion_loss(self, mu_a, r_a, mu_b, r_b) -> torch.Tensor:
        """
        互斥排斥势能 (Disjoint Hinge Repulsion for A \perp B):
        max(0, R_A + R_B + delta_rep - ||mu_A - mu_B||_2)
        """
        dist = torch.norm(mu_a - mu_b, p=2, dim=-1)
        violation = F.relu(r_a + r_b + self.delta_rep - dist)
        return violation.mean()

    def surface_tension_loss(self, all_radii: torch.Tensor) -> torch.Tensor:
        """
        表面张力与体积守恒惩罚 (Surface Tension / Anti-Volume Exploitation):
        防止没有子概念支撑的节点无限制膨胀，将叶子节点压缩至 Level 1。
        """
        return torch.mean(all_radii ** 2)

    def phase1_emergence_loss(self, manifold, sub_idx, super_idx, ant_a_idx, ant_b_idx):
        """Phase 1 专用：纯物理势能涌现总损失"""
        mu_sub = manifold.mu[sub_idx]
        r_sub = manifold.get_radius(sub_idx)
        mu_sup = manifold.mu[super_idx]
        r_sup = manifold.get_radius(super_idx)
        l_inc = self.inclusion_loss(mu_sub, r_sub, mu_sup, r_sup)

        if len(ant_a_idx) > 0 and len(ant_b_idx) > 0:
            mu_a = manifold.mu[ant_a_idx]
            r_a = manifold.get_radius(ant_a_idx)
            mu_b = manifold.mu[ant_b_idx]
            r_b = manifold.get_radius(ant_b_idx)
            l_rep = self.repulsion_loss(mu_a, r_a, mu_b, r_b)
        else:
            l_rep = torch.zeros_like(l_inc)

        l_tension = self.surface_tension_loss(manifold.get_radius())
        total_phys = self.beta_rel * (l_inc + l_rep) + self.alpha_tension * l_tension
        return total_phys, {"L_inc": l_inc.item(), "L_rep": l_rep.item(), "L_tension": l_tension.item()}
