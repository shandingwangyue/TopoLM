import torch
import torch.nn as nn
import torch.nn.functional as F

class TopoDomainSpace(nn.Module):
    def __init__(self, num_concepts, dim, tau=0.1, margin=1.0):
        super().__init__()
        self.dim = dim
        self.tau = tau
        self.margin = margin
        
        # 概念域的中心向量 (u_c) 和 半径 (r_c)
        self.centers = nn.Parameter(torch.randn(num_concepts, dim))
        self.radii = nn.Parameter(torch.ones(num_concepts) * 0.5)
        
    def membership_score(self, x_emb, concept_idx):
        """计算Token嵌入 x_emb 在指定概念中的成员资格分数"""
        u_c = self.centers[concept_idx]
        r_c = F.softplus(self.radii[concept_idx]) # 保证半径为正
        dist = torch.norm(x_emb - u_c, p=2, dim=-1)
        # 依据公式: m(x, c) = sigmoid((r_c - ||e(x) - u_c||) / tau)
        score = torch.sigmoid((r_c - dist) / self.tau)
        return score

    def inclusion_loss(self, sub_idx, super_idx):
        """计算 c1 包含于 c2 的结构损失 (L_rel 的一部分)"""
        u_1, u_2 = self.centers[sub_idx], self.centers[super_idx]
        r_1 = F.softplus(self.radii[sub_idx])
        r_2 = F.softplus(self.radii[super_idx])
        
        dist = torch.norm(u_1 - u_2, p=2)
        # 理想情况下: dist + r_1 <= r_2
        violation = F.relu(dist + r_1 - r_2) 
        return violation

    def disjoint_loss(self, c1_idx, c2_idx):
        """计算 c1 与 c2 互斥的结构损失 (L_rel 的另一部分)"""
        u_1, u_2 = self.centers[c1_idx], self.centers[c2_idx]
        r_1 = F.softplus(self.radii[c1_idx])
        r_2 = F.softplus(self.radii[c2_idx])
        
        dist = torch.norm(u_1 - u_2, p=2)
        # 理想情况下: dist >= r_1 + r_2 + delta
        violation = F.relu(r_1 + r_2 + self.margin - dist)
        return violation

# --- 测试验证代码 ---
if __name__ == "__main__":
    # 初始化拓扑空间，假设有3个概念：0=Animal(领域带), 1=Cat(实体带), 2=Machine(领域带)
    topo_space = TopoDomainSpace(num_concepts=3, dim=256)
    
    # 模拟构建损失函数的过程: L = L_LM + alpha*L_mem + beta*L_rel + gamma*L_con
    
    # 1. 设置知识图谱/本体约束
    loss_inc = topo_space.inclusion_loss(sub_idx=1, super_idx=0) # Cat 必须属于 Animal
    loss_dis = topo_space.disjoint_loss(c1_idx=0, c2_idx=2)      # Animal 必须与 Machine 互斥
    
    # 2. 模拟Token成员资格检测
    mock_token_emb = torch.randn(256) 
    score_cat = topo_space.membership_score(mock_token_emb, concept_idx=1)
    
    # 整体结构损失
    beta = 1.0
    L_rel = beta * (loss_inc + loss_dis)
    print(f"当前未优化的结构矛盾损失 (L_rel): {L_rel.item():.4f}")
