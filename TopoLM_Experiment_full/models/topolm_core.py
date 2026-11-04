import torch
import torch.nn as nn
from models.topological_attention import AnalyticalWassersteinAttention

class TopoLM(nn.Module):
    def __init__(self, vocab_size, hidden_dim, num_heads, num_layers):
        super().__init__()
        self.vocab_size = vocab_size  # 保存词表规模
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.layers = nn.ModuleList([
            AnalyticalWassersteinAttention(hidden_dim, num_heads, tau=0.5)
            for _ in range(num_layers)
        ])
        self.lm_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        # 【认知护盾 1】：绝对输入钳制，杜绝因梯度毒性导致的游离 Token 越界
        x = torch.clamp(x, 0, self.vocab_size - 1)
        
        h = self.embedding(x)
        mu_out, logvar_out = None, None
        
        for layer in self.layers:
            h, mu_out, logvar_out = layer(h)
            
        # 【认知护盾 2】：能量缩放 (Energy Scaling)
        # 除以隐藏维度的平方根，防止 Phase 2 启动瞬间未经训练的 lm_head 发生熵爆炸
        logits = self.lm_head(h) / (h.size(-1) ** 0.5)
        
        return logits, mu_out, logvar_out

    @torch.no_grad()
    def extract_9_level_stratification(self, logvar):
        domain_volumes = torch.sum(logvar, dim=-1) 
        quantiles = torch.quantile(
            domain_volumes, 
            torch.linspace(0.11, 1.0, 9, device=logvar.device)
        )
        return torch.bucketize(domain_volumes, quantiles) + 1