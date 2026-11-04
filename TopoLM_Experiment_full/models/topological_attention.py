import torch
import torch.nn as nn
import torch.nn.functional as F

class TopologicalDomainEmbedding(nn.Module):
    def __init__(self, hidden_dim, domain_dim):
        super().__init__()
        self.fc_mu = nn.Linear(hidden_dim, domain_dim)
        self.fc_logvar = nn.Linear(hidden_dim, domain_dim)
        # 初始化合理的域边界
        nn.init.normal_(self.fc_logvar.weight, mean=0.0, std=0.01)

    def forward(self, x):
        return self.fc_mu(x), self.fc_logvar(x)

class AnalyticalWassersteinAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, tau=1.0):
        super().__init__()
        self.num_heads = num_heads
        self.domain_dim = hidden_dim // num_heads
        self.tau = tau
        
        self.q_domain = TopologicalDomainEmbedding(hidden_dim, hidden_dim)
        self.k_domain = TopologicalDomainEmbedding(hidden_dim, hidden_dim)
        self.v_domain = TopologicalDomainEmbedding(hidden_dim, hidden_dim)

    def compute_pairwise_sq_dist(self, x, y):
        x_sq = torch.sum(x ** 2, dim=-1, keepdim=True)
        y_sq = torch.sum(y ** 2, dim=-1, keepdim=True).transpose(-2, -1)
        xy = torch.matmul(x, y.transpose(-2, -1))
        return torch.clamp(x_sq + y_sq - 2 * xy, min=1e-8)

    def forward(self, x):
        B, L, _ = x.size()
        reshape_fn = lambda t: t.view(B, L, self.num_heads, self.domain_dim).transpose(1, 2)
        
        q_mu, q_logvar = reshape_fn(self.q_domain.fc_mu(x)), reshape_fn(self.q_domain.fc_logvar(x))
        k_mu, k_logvar = reshape_fn(self.k_domain.fc_mu(x)), reshape_fn(self.k_domain.fc_logvar(x))
        v_mu = reshape_fn(self.v_domain.fc_mu(x))
        
        q_sigma = torch.exp(0.5 * q_logvar)
        k_sigma = torch.exp(0.5 * k_logvar)

        # 2-Wasserstein 解析几何距离
        w2_sq_dist = self.compute_pairwise_sq_dist(q_mu, k_mu) + self.compute_pairwise_sq_dist(q_sigma, k_sigma)

        # 转换为注意力分数
        scores = -w2_sq_dist / (self.tau * (self.domain_dim ** 0.5))
        attn_weights = F.softmax(scores, dim=-1)

        output = torch.matmul(attn_weights, v_mu)
        output = output.transpose(1, 2).contiguous().view(B, L, -1)
        
        # 返回全局拓扑参数供能量泛函调用
        global_mu = self.q_domain.fc_mu(x)
        global_logvar = self.q_domain.fc_logvar(x)
        
        return output, global_mu, global_logvar