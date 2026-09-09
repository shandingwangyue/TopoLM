import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math

# ==========================================
# 1. 核心网络模型 (含数值稳定性修复)
# ==========================================
class TopologicalDomainSpace(nn.Module):
    def __init__(self, num_domains, d_model):
        super().__init__()
        self.num_domains = num_domains
        self.d_model = d_model
        # 使用 Xavier 初始化中心点，让初始距离不至于太离谱
        self.mu = nn.Parameter(torch.randn(num_domains, d_model) * 0.1)
        self.log_sigma = nn.Parameter(torch.zeros(num_domains, d_model))
        
    def get_sigma(self):
        return torch.exp(self.log_sigma)

    def compute_inclusion_prob(self, x, domain_idx):
        mu_d = self.mu[domain_idx]
        sigma_d = self.get_sigma()[domain_idx]
        
        diff = x - mu_d
        mahalanobis_sq = (diff ** 2) / (sigma_d + 1e-9)
        distance = mahalanobis_sq.sum(dim=-1)
        
        prob = torch.exp(-0.5 * distance)
        return prob

class SpatialIndexedAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, topo_space, active_domain_idx, threshold=0.01):
        batch_size, seq_len, _ = x.size()
        
        inclusion_prob = topo_space.compute_inclusion_prob(x, active_domain_idx)
        domain_mask = (inclusion_prob >= threshold).float()
        attn_mask = domain_mask.unsqueeze(1).unsqueeze(2)
        
        qkv = self.qkv_proj(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2), qkv)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        
        # 【关键修复】使用 -1e4 替代 float('-inf')，防止前期随机初始化导致全部被 Mask 产生的 NaN 崩溃
        scores = scores.masked_fill(attn_mask == 0, -1e4)
        attn_weights = F.softmax(scores, dim=-1)
        
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        return self.out_proj(out), inclusion_prob

class TopoLM(nn.Module):
    def __init__(self, vocab_size, num_domains, d_model, num_heads):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.topo_space = TopologicalDomainSpace(num_domains, d_model)
        self.spatial_attn = SpatialIndexedAttention(d_model, num_heads)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, active_domain_idx):
        x = self.token_emb(input_ids)
        attn_out, probs = self.spatial_attn(x, self.topo_space, active_domain_idx)
        logits = self.lm_head(attn_out)
        return logits, probs

# ==========================================
# 2. 联合优化损失函数
# ==========================================
class TopoLMLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=1.0, margin=5.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.margin = margin
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, logits, targets, probs, active_domain_idx, topo_space, disjoint_pairs=[]):
        l_lm = self.ce_loss(logits.view(-1, logits.size(-1)), targets.view(-1))
        # 限制下界，防止对 0 取 log
        l_mem = -torch.log(probs.clamp(min=1e-6)).mean()
        
        l_rel = torch.tensor(0.0, device=logits.device)
        for d1, d2 in disjoint_pairs:
            mu1, mu2 = topo_space.mu[d1], topo_space.mu[d2]
            dist = torch.norm(mu1 - mu2, p=2)
            # Hinge loss: 如果距离小于安全 margin，则严厉惩罚
            l_rel += F.relu(self.margin - dist)

        total_loss = l_lm + self.alpha * l_mem + self.beta * l_rel
        return total_loss, l_lm, l_mem, l_rel

# ==========================================
# 3. 实验运行逻辑
# ==========================================
def run_experiment():
    torch.manual_seed(42)
    
    VOCAB_SIZE = 100
    NUM_DOMAINS = 5
    D_MODEL = 32
    NUM_HEADS = 2
    BATCH_SIZE = 4
    SEQ_LEN = 8
    EPOCHS = 101 # 跑100轮看趋势

    model = TopoLM(VOCAB_SIZE, NUM_DOMAINS, D_MODEL, NUM_HEADS)
    
    # 我们希望域之间的距离至少拉开 8.0
    criterion = TopoLMLoss(alpha=0.2, beta=0.5, margin=8.0) 
    
    DOMAIN_PHYSICS = 0
    DOMAIN_FANTASY = 1
    disjoint_pairs = [(DOMAIN_PHYSICS, DOMAIN_FANTASY)]
    
    optimizer = optim.AdamW(model.parameters(), lr=0.01)

    print("=== 开始训练验证 TopoLM 拓扑约束 ===\n")
    
    for epoch in range(EPOCHS):
        input_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
        targets = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))
        
        optimizer.zero_grad()
        logits, probs = model(input_ids, active_domain_idx=DOMAIN_PHYSICS)
        
        loss, l_lm, l_mem, l_rel = criterion(
            logits, targets, probs, 
            active_domain_idx=DOMAIN_PHYSICS,
            topo_space=model.topo_space,
            disjoint_pairs=disjoint_pairs
        )
        
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            mu_phys = model.topo_space.mu[DOMAIN_PHYSICS]
            mu_fant = model.topo_space.mu[DOMAIN_FANTASY]
            dist = torch.norm(mu_phys - mu_fant, p=2).item()
            avg_prob = probs.mean().item()
            
            print(f"Epoch {epoch:03d} | Total Loss: {loss.item():.4f} (LM: {l_lm.item():.2f}, Mem: {l_mem.item():.2f}, Rel: {l_rel.item():.2f})")
            print(f"  -> 平均包含概率 (Inclusion Prob): {avg_prob:.4f}")
            print(f"  -> 物理域与奇幻域的几何距离:      {dist:.4f}\n")

if __name__ == "__main__":
    run_experiment()