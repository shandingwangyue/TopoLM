import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TopoDomainSpace(nn.Module):
    """
    Manages the 9-Level Topological Domains using Multivariate Gaussian Distributions.
    """
    def __init__(self, num_domains, d_model, margin=2.0):
        super().__init__()
        self.d_model = d_model
        self.margin = margin
        
        # Domain Centroids (mu) and Diagonal Covariance (log_sigma for stability)
        self.mu = nn.Parameter(torch.randn(num_domains, d_model))
        self.log_sigma = nn.Parameter(torch.zeros(num_domains, d_model))
        
    def get_sigma(self):
        # Convert log_sigma to positive variance
        return torch.exp(self.log_sigma)

    def compute_pdf(self, x, domain_idx):
        """
        Calculates the inclusion probability density of vector x in a specific domain.
        x: [batch_size, seq_len, d_model]
        """
        mu_d = self.mu[domain_idx]
        sigma_d = self.get_sigma()[domain_idx]
        
        # Mahalanobis distance squared
        diff = x - mu_d
        mahalanobis_sq = (diff ** 2) / (sigma_d + 1e-9)
        distance = mahalanobis_sq.sum(dim=-1)
        
        # Probability Density calculation
        prob = torch.exp(-0.5 * distance)
        return prob

    def kl_divergence_loss(self, sub_idx, super_idx):
        """
        Calculates KL Divergence for Structural Inclusion (Layer 2 within Layer 1).
        """
        mu_1, mu_2 = self.mu[sub_idx], self.mu[super_idx]
        sig_1, sig_2 = self.get_sigma()[sub_idx], self.get_sigma()[super_idx]
        
        term1 = (sig_1 / (sig_2 + 1e-9)).sum()
        term2 = ((mu_2 - mu_1) ** 2 / (sig_2 + 1e-9)).sum()
        term3 = self.d_model
        term4 = torch.log((sig_2.prod() + 1e-9) / (sig_1.prod() + 1e-9))
        
        kl_div = 0.5 * (term1 + term2 - term3 + term4)
        return kl_div

    def disjoint_loss(self, idx1, idx2):
        """
        Calculates Hinge Loss for Mutual Exclusivity (e.g., perpetual motion vs physics).
        """
        mu_1, mu_2 = self.mu[idx1], self.mu[idx2]
        dist = torch.norm(mu_1 - mu_2, p=2)
        # Margin-based disjoint enforcement
        return F.relu(self.margin - dist)


class SpatialIndexedAttention(nn.Module):
    """
    Executes Localized Attention masked by Topological Domain constraints.
    """
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, topo_space, active_domain_idx, threshold=0.01):
        batch_size, seq_len, _ = x.size()
        
        # Step 1: Compute Topological Inclusion Mask
        inclusion_prob = topo_space.compute_pdf(x, active_domain_idx)
        domain_mask = (inclusion_prob >= threshold).float()
        attn_mask = domain_mask.unsqueeze(1).unsqueeze(2) # [B, 1, 1, S]
        
        # Step 2: QKV Projection
        qkv = self.qkv_proj(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2), qkv)
        
        # Step 3: Spatially Masked Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        scores = scores.masked_fill(attn_mask == 0, float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)
        
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        return self.out_proj(out), inclusion_prob


# --- Validation and Use Cases ---
if __name__ == "__main__":
    torch.manual_seed(42)
    d_model = 256
    
    # Initialize Architecture
    topo_space = TopoDomainSpace(num_domains=10, d_model=d_model)
    attn_layer = SpatialIndexedAttention(d_model, num_heads=8)
    
    # Define Indices
    DOMAIN_PHYSICS = 0  # Macro Domain (Layer 1)
    DOMAIN_XIANXIA = 1  # Meso Domain (Layer 2)
    
    print("=== TopoLM Execution Trace ===")
    
    # Case 1: Axiomatic Violation (Thermodynamics)
    print("\n[Case 1] Input: 'Closed-system 100% efficiency engine'")
    tokens_physics = torch.cat([
        torch.randn(1, 1, d_model) * 0.1,  # "engine" (In domain)
        torch.randn(1, 1, d_model) * 5.0   # "perpetual_motion" (Out of bounds)
    ], dim=1)
    
    _, probs_physics = attn_layer(tokens_physics, topo_space, DOMAIN_PHYSICS)
    
    for i, token in enumerate(["engine", "perpetual_motion"]):
        prob = probs_physics[0][i].item()
        status = "ACCEPTED" if prob >= 0.01 else "REJECTED (Computational Friction)"
        print(f"Token: {token:<20} | Prob: {prob:.4f} | {status}")
        
    # Case 2: Cross-Domain Decoupling (Literary Creation)
    print("\n[Case 2] Input: 'Ascending to the clouds without a sword' (《十万里风雪客》)")
    tokens_xianxia = torch.cat([
        torch.randn(1, 1, d_model) * 0.1,  # "inner_qi"
        torch.randn(1, 1, d_model) * 0.1   # "ascension"
    ], dim=1)
    
    # Shift centroid to simulate switching to the Xianxia domain
    topo_space.mu.data[DOMAIN_XIANXIA] = tokens_xianxia[0].mean(dim=0)
    
    _, probs_xianxia = attn_layer(tokens_xianxia, topo_space, DOMAIN_XIANXIA)
    
    for i, token in enumerate(["inner_qi", "ascension"]):
        prob = probs_xianxia[0][i].item()
        status = "ACCEPTED" if prob >= 0.01 else "REJECTED"
        print(f"Token: {token:<20} | Prob: {prob:.4f} | {status}")