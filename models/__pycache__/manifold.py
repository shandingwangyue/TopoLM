import torch
import torch.nn as nn
import torch.nn.functional as F

class TopologicalDomainEmbedding(nn.Module):
    """
    统一拓扑域嵌入层 (Unified Topological Domain Embedding):
    将每个 Token/概念 建模为 d 维流形上的多维高斯分布 D_i ~ N(mu_i, Sigma_i) 或有界超球体。
    严格对应论文 Section 4 公式 (1)-(4)。
    """
    def __init__(self, num_concepts: int, d_model: int = 128, sigma_min: float = 0.05, init_sigma_range=(0.19, 0.32)):
        super().__init__()
        self.num_concepts = num_concepts
        self.d_model = d_model
        self.sigma_min = sigma_min

        # 语义质心 mu_i \in R^d
        self.mu = nn.Parameter(torch.randn(num_concepts, d_model) * 0.1)
        
        # 初始半径控制在微米级 [0.19, 0.32]，通过逆 softplus 参数化确保正定
        init_radii = torch.empty(num_concepts).uniform_(*init_sigma_range)
        inv_softplus = torch.log(torch.exp(init_radii - sigma_min) - 1.0)
        self.raw_sigma = nn.Parameter(inv_softplus)

    @property
    def sigma(self) -> torch.Tensor:
        """物理半径 R_i / 标准差 sigma_i = softplus(raw_sigma) + sigma_min"""
        return F.softplus(self.raw_sigma) + self.sigma_min

    def freeze_backbone(self):
        """Phase 2 零幻觉协议：永久冻结底层拓扑坐标与半径 (nabla_{mu, sigma} = 0)"""
        self.mu.requires_grad = False
        self.raw_sigma.requires_grad = False

    def containment_margin(self, sub_idx: torch.Tensor, super_idx: torch.Tensor) -> torch.Tensor:
        """
        计算非对称包含裕度 (Containment Margin):
        Delta_margin(D_sub, D_super) = R_super - R_sub - ||mu_sub - mu_super||_2
        大于 0 表示 D_sub 严格嵌套在 D_super 内部 (A \subset B)。
        """
        mu_sub, mu_super = self.mu[sub_idx], self.mu[super_idx]
        r_sub, r_super = self.sigma[sub_idx], self.sigma[super_idx]
        dist = torch.norm(mu_sub - mu_super, p=2, dim=-1)
        return r_super - r_sub - dist

    def mahalanobis_energy(self, x: torch.Tensor, domain_idx: torch.Tensor) -> torch.Tensor:
        """
        计算实体向量 x 相对于目标拓扑域 D_i 的平方马氏违背能量 d_M^2(x, D_i)
        对应论文公式 (2) 与 Experiment E3。
        """
        mu_d = self.mu[domain_idx]
        var_d = (self.sigma[domain_idx] ** 2).unsqueeze(-1)
        diff = x - mu_d
        return torch.sum((diff ** 2) / (var_d + 1e-8), dim=-1)

    def forward(self, token_ids: torch.Tensor):
        """返回对应 Token 的质心向量 mu 与当前半径 sigma"""
        return self.mu[token_ids], self.sigma[token_ids]