"""
TopoLM: Topological Domain Space Manager
File: models/domain_space.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TopologicalDomainSpace(nn.Module):
    """
    负责维护所有层级语义域（Domains）的高维多元高斯几何参数。
    支持对角协方差求值、无损马氏距离运算与几何包含裕度评估。
    """
    def __init__(self, num_domains: int, d_model: int, init_log_sigma: float = 0.5):
        super().__init__()
        self.num_domains = num_domains
        self.d_model = d_model
        
        # 语义中心 mu 与对数方差 log_sigma (防止方差优化出现负值)
        self.mu = nn.Parameter(torch.randn(num_domains, d_model) * 0.1)
        self.log_sigma = nn.Parameter(torch.full((num_domains, d_model), init_log_sigma))

    def get_sigma(self) -> torch.Tensor:
        """获取裁剪至稳定区间的正方差"""
        return torch.exp(self.log_sigma).clamp(min=1e-4, max=50.0)

    def mahalanobis_sq(self, x: torch.Tensor, domain_idx: int) -> torch.Tensor:
        """
        计算输入向量到指定域的马氏距离平方: (x - mu)^T Sigma^-1 (x - mu)
        Args:
            x: [..., d_model] 语义向量
            domain_idx: 目标域索引
        Returns:
            dist_sq: [...] 与输入前导维度一致的距离标量
        """
        mu = self.mu[domain_idx]
        sigma = self.get_sigma()[domain_idx]
        diff = x - mu
        return (((diff ** 2) / sigma)).sum(dim=-1)

    def compute_all_distances(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算输入序列到所有域的马氏距离平方矩阵。
        Args:
            x: [batch, seq_len, d_model]
        Returns:
            all_dists: [num_domains, batch, seq_len]
        """
        dists = [self.mahalanobis_sq(x, d) for d in range(self.num_domains)]
        return torch.stack(dists, dim=0)

    def membership_prob(self, x: torch.Tensor, domain_idx: int) -> torch.Tensor:
        """计算高斯概率密度隶属度得分 (0.0 到 1.0)"""
        dist_sq = self.mahalanobis_sq(x, domain_idx)
        return torch.exp(-0.5 * dist_sq)

    def domain_centroid_distance(self, d1: int, d2: int) -> torch.Tensor:
        """计算两个域中心之间的欧氏距离"""
        return torch.norm(self.mu[d1] - self.mu[d2], p=2)

    def containment_margin(self, sub_idx: int, super_idx: int) -> torch.Tensor:
        """
        计算子域外包络是否落入父域内: R_super - R_sub - Dist(u_sub, u_super)
        """
        u_sub, u_super = self.mu[sub_idx], self.mu[super_idx]
        sig_sub, sig_super = self.get_sigma()[sub_idx], self.get_sigma()[super_idx]
        
        dist = torch.norm(u_sub - u_super, p=2)
        r_sub = torch.sqrt(sig_sub.sum())
        r_super = torch.sqrt(sig_super.sum())
        return r_super - r_sub - dist