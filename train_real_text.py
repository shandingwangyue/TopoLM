import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from transformers import BertTokenizer

# ==========================================
# 1. 核心网络模型 (含马氏距离梯度修复)
# ==========================================
class TopologicalDomainSpace(nn.Module):
    def __init__(self, num_domains, d_model):
        super().__init__()
        self.num_domains = num_domains
        self.d_model = d_model
        self.mu = nn.Parameter(torch.randn(num_domains, d_model) * 0.1)
        self.log_sigma = nn.Parameter(torch.ones(num_domains, d_model) * 2.0)
        
    def get_sigma(self):
        return torch.exp(self.log_sigma)

    def compute_inclusion_prob(self, x, domain_idx):
        mu_d = self.mu[domain_idx]
        sigma_d = self.get_sigma()[domain_idx]
        diff = x - mu_d
        mahalanobis_sq = (diff ** 2) / (sigma_d + 1e-9)
        distance = mahalanobis_sq.sum(dim=-1)
        prob = torch.exp(-0.5 * distance)
        # 抛出 distance 用于无损计算梯度
        return prob, distance

class SpatialIndexedAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    # 【微调 1】将解封阈值降低至 1e-4，让它在训练中后期能够顺利解开掩码
    def forward(self, x, topo_space, active_domain_idx, threshold=1e-9):
        batch_size, seq_len, _ = x.size()
        
        inclusion_prob, distance = topo_space.compute_inclusion_prob(x, active_domain_idx)
        domain_mask = (inclusion_prob >= threshold).float()
        attn_mask = domain_mask.unsqueeze(1).unsqueeze(2) # [B, 1, 1, S]
        
        # 【微调 2】引入下三角因果掩码 (Causal Mask)，强制自回归模型只能看过去
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).view(1, 1, seq_len, seq_len)
        
        qkv = self.qkv_proj(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2), qkv)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        
        # 叠加因果掩码与拓扑掩码
        scores = scores.masked_fill(causal_mask == 0, -1e4)
        scores = scores.masked_fill(attn_mask == 0, -1e4)
        
        attn_weights = F.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        return self.out_proj(out), inclusion_prob, distance

class TopoLM(nn.Module):
    def __init__(self, vocab_size, num_domains, d_model, num_heads, max_seq_len=128):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        # 【微调 3】引入绝对位置编码 (Positional Embeddings)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        
        self.topo_space = TopologicalDomainSpace(num_domains, d_model)
        self.spatial_attn = SpatialIndexedAttention(d_model, num_heads)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, active_domain_idx):
        seq_len = input_ids.size(1)
        # 生成位置索引
        positions = torch.arange(0, seq_len, device=input_ids.device).unsqueeze(0)
        
        # Token 嵌入 + 位置嵌入
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        
        attn_out, probs, distance = self.spatial_attn(x, self.topo_space, active_domain_idx)
        logits = self.lm_head(attn_out)
        return logits, probs, distance

# ==========================================
# 2. 联合优化损失函数 (无截断优化)
# ==========================================
class TopoLMLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=1.0, margin=5.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.margin = margin
        self.ce_loss = nn.CrossEntropyLoss()

    # 接收 distance 参数
    def forward(self, logits, targets, probs, distance, active_domain_idx, topo_space, disjoint_pairs=[]):
        l_lm = self.ce_loss(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        # 直接利用马氏距离进行惩罚，彻底消除 log(0) 带来的梯度截断
        l_mem = 0.5 * distance.mean()
        
        l_rel = torch.tensor(0.0, device=logits.device)
        for d1, d2 in disjoint_pairs:
            mu1, mu2 = topo_space.mu[d1], topo_space.mu[d2]
            dist = torch.norm(mu1 - mu2, p=2)
            l_rel += F.relu(self.margin - dist)

        total_loss = l_lm + self.alpha * l_mem + self.beta * l_rel
        return total_loss, l_lm, l_mem, l_rel

# ==========================================
# 3. 真实语料训练流 (数据预处理与循环)
# ==========================================
def create_dataset(tokenizer, text, seq_len):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    inputs, targets = [], []
    for i in range(0, len(tokens) - seq_len):
        inputs.append(tokens[i : i + seq_len])
        targets.append(tokens[i + 1 : i + seq_len + 1])
    return torch.tensor(inputs), torch.tensor(targets)

def run_real_text_experiment():
    torch.manual_seed(42)
    print("正在加载 Tokenizer (bert-base-chinese)...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    VOCAB_SIZE = tokenizer.vocab_size
    
    NUM_DOMAINS = 2
    D_MODEL = 64
    NUM_HEADS = 4
    SEQ_LEN = 12
    EPOCHS = 150
    
    DOMAIN_PHYSICS = 0
    DOMAIN_FANTASY = 1
    disjoint_pairs = [(DOMAIN_PHYSICS, DOMAIN_FANTASY)]

    text_physics = "孤立系统的熵在不伴随外界作功的情况下永远不会减少，热力学定律是物理学的绝对法则。"
    text_fantasy = "十万里风雪客运转真气，踏破虚空，不借助飞剑便直上云霄，这是独属于他的仙侠大道。"
    
    x_phys, y_phys = create_dataset(tokenizer, text_physics, SEQ_LEN)
    x_fant, y_fant = create_dataset(tokenizer, text_fantasy, SEQ_LEN)

    model = TopoLM(VOCAB_SIZE, NUM_DOMAINS, D_MODEL, NUM_HEADS)
    
    # 调大 alpha 使成员资格优化具有足够主导权拉拢 Token
    criterion = TopoLMLoss(alpha=0.5, beta=0.8, margin=8.0) 
    optimizer = optim.AdamW(model.parameters(), lr=5e-3)

    print(f"\n=== 开始在真实语料上训练 TopoLM (词表大小: {VOCAB_SIZE}) ===\n")
    
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        
        # --- 物理域前向与损失计算 ---
        # 接收三项输出
        logits_p, probs_p, dist_p = model(x_phys, active_domain_idx=DOMAIN_PHYSICS)
        # 传入 dist_p
        loss_p, lm_p, mem_p, rel_p = criterion(
            logits_p, y_phys, probs_p, dist_p, DOMAIN_PHYSICS, model.topo_space, disjoint_pairs
        )
        
        # --- 仙侠域前向与损失计算 ---
        # 接收三项输出
        logits_f, probs_f, dist_f = model(x_fant, active_domain_idx=DOMAIN_FANTASY)
        # 传入 dist_f
        loss_f, lm_f, mem_f, rel_f = criterion(
            logits_f, y_fant, probs_f, dist_f, DOMAIN_FANTASY, model.topo_space, disjoint_pairs
        )
        
        total_loss = loss_p + loss_f
        total_loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0 or epoch == EPOCHS - 1:
            dist = torch.norm(model.topo_space.mu[0] - model.topo_space.mu[1], p=2).item()
            
            print(f"Epoch {epoch:03d} | 物理域 L_LM: {lm_p.item():.4f} | 仙侠域 L_LM: {lm_f.item():.4f}")
            print(f"            | 平均 L_mem: {(mem_p + mem_f).item()/2:.4f} | 结构互斥 L_rel: {rel_p.item():.4f}")
            print(f"            | 拓扑距离 (物理 <-> 仙侠): {dist:.4f}")
            print(f"            | 物理域平均包含概率: {probs_p.mean().item():.4f}\n")
            
            if epoch == EPOCHS - 1:
                print("--- 语言生成验证 ---")
                sample_input = x_fant[0:1]
                prompt_text = tokenizer.decode(sample_input[0].tolist())
                
                # 推理时同样接收三个返回值，忽略距离输出即可
                logits_test, _, _ = model(sample_input, DOMAIN_FANTASY)
                pred_token_ids = torch.argmax(logits_test[0], dim=-1)
                pred_text = tokenizer.decode(pred_token_ids.tolist())
                
                print(f"Prompt (仙侠域): {prompt_text}")
                print(f"模型预测 Next-Tokens: {pred_text}")

if __name__ == "__main__":
    run_real_text_experiment()