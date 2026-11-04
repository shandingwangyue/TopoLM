import torch
import torch.nn as nn
import torch.nn.functional as F

class TopologicalEnergyLoss(nn.Module):
    def __init__(self, margin=2.0, volume_reg_weight=0.01):
        super().__init__()
        self.margin = margin
        self.volume_reg = volume_reg_weight

    def forward(self, mu_A, logvar_A, mu_B, logvar_B, relations):
        # 将 logvar 转换为实际半径，加上 0.1 基础体积保证微观粒子下限
        radius_A = F.softplus(logvar_A.mean(dim=-1, keepdim=True)) + 0.1
        radius_B = F.softplus(logvar_B.mean(dim=-1, keepdim=True)) + 0.1
        
        dist_mu = torch.norm(mu_A - mu_B, dim=-1, keepdim=True)
        
        # 1. 结构嵌套律 (A属于B -> 关系=1)
        # B必须膨胀来包裹A，强迫产生层级差
        inclusion_loss = dist_mu + F.relu(radius_A + 0.5 - radius_B)
        
        # 2. 空间排斥律 (互斥 -> 关系=-1)
        exclusion_loss = F.relu(radius_A + radius_B + 2.0 - dist_mu)
        
        # 3. 体积张力惩罚
        volume_penalty = 0.01 * (radius_A ** 2 + radius_B ** 2)
        
        loss_pull = torch.where(relations > 0.5, inclusion_loss, torch.zeros_like(inclusion_loss))
        loss_push = torch.where(relations < -0.5, exclusion_loss, torch.zeros_like(exclusion_loss))

        return loss_pull.mean() + loss_push.mean() + volume_penalty.mean()