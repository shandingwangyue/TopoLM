"""
TopoLM: Topological Domain Space Manager
File: models/domain_space.py
"""

import torch
import torch.nn as nn
import math


class TopologicalDomainSpace(nn.Module):
    def __init__(self, num_domains: int, d_model: int, init_log_sigma: float = 0.5):
        super().__init__()
        if num_domains < 1 or d_model < 1:
            raise ValueError("num_domains and d_model must be positive")
        self.num_domains = num_domains
        self.d_model = d_model
        
        self.mu = nn.Parameter(torch.randn(num_domains, d_model) * 0.1)
        self.log_sigma = nn.Parameter(torch.full((num_domains, d_model), init_log_sigma))

    def get_sigma(self) -> torch.Tensor:
        return self.get_log_variance().exp()

    def get_log_variance(self) -> torch.Tensor:
        return self.log_sigma.clamp(math.log(1e-4), math.log(50.0))

    def get_radius(self, domain_idx=None) -> torch.Tensor:
        sig = self.get_sigma()
        if domain_idx is not None:
            return torch.sqrt(sig[domain_idx].sum(dim=-1))
        return torch.sqrt(sig.sum(dim=-1))

    def freeze_backbone(self):
        self.mu.requires_grad = False
        self.log_sigma.requires_grad = False

    def mahalanobis_sq(self, x: torch.Tensor, domain_idx: int) -> torch.Tensor:
        if not 0 <= domain_idx < self.num_domains:
            raise ValueError(f"Invalid domain index: {domain_idx}")
        mu = self.mu[domain_idx]
        sigma = self.get_sigma()[domain_idx]
        diff = x - mu
        return (((diff ** 2) / sigma)).sum(dim=-1)

    def compute_all_distances(self, x: torch.Tensor) -> torch.Tensor:
        dists = [self.mahalanobis_sq(x, d) for d in range(self.num_domains)]
        return torch.stack(dists, dim=0)

    def membership_prob(self, x: torch.Tensor, domain_idx: int) -> torch.Tensor:
        dist_sq = self.mahalanobis_sq(x, domain_idx)
        return torch.exp(-0.5 * dist_sq)

    def domain_centroid_distance(self, d1: int, d2: int) -> torch.Tensor:
        return torch.norm(self.mu[d1] - self.mu[d2], p=2)

    def containment_margin(self, sub_idx: int, super_idx: int) -> torch.Tensor:
        u_sub, u_super = self.mu[sub_idx], self.mu[super_idx]
        sig_sub, sig_super = self.get_sigma()[sub_idx], self.get_sigma()[super_idx]
        dist = torch.norm(u_sub - u_super, p=2)
        r_sub = torch.sqrt(sig_sub.sum())
        r_super = torch.sqrt(sig_super.sum())
        return r_super - r_sub - dist
