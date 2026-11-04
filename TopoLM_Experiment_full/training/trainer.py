import torch
import torch.nn as nn
import torch.optim as optim

def train_epoch(model, dataloader, topo_criterion, lm_criterion, optimizer, device):
    model.train()
    total_loss, total_topo, total_lm = 0, 0, 0
    phase = dataloader.dataset.phase
    
    for context, relations in dataloader:
        context, relations = context.to(device), relations.to(device)
        optimizer.zero_grad()
        
        logits, mu, logvar = model(context)
        
        # 提取 Anchor 和 Target 域
        mu_anchor, logvar_anchor = mu[:, 0, :], logvar[:, 0, :]
        mu_target, logvar_target = mu[:, 1, :], logvar[:, 1, :]
        
        loss_topo = topo_criterion(mu_anchor, logvar_anchor, mu_target, logvar_target, relations)
        
        loss_lm = torch.tensor(0.0, device=device)
        if phase == 'unsupervised':
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = context[:, 1:].contiguous()
            
            # 【认知护盾 3】：目标端钳制，保护交叉熵核函数绝对安全
            shift_labels = torch.clamp(shift_labels, 0, model.vocab_size - 1)
            
            loss_lm = lm_criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
        loss = loss_topo + loss_lm
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        total_topo += loss_topo.item()
        if phase == 'unsupervised': total_lm += loss_lm.item()
        
    return total_loss / len(dataloader), total_topo / len(dataloader), total_lm / len(dataloader)

def monitor_stratification(model, dataloader, device):
    model.eval()
    with torch.no_grad():
        sample_context, _ = next(iter(dataloader))
        sample_context = sample_context.to(device)
        _, _, logvar = model(sample_context)
        
        levels = model.extract_9_level_stratification(logvar)
        counts = torch.bincount(levels.view(-1), minlength=10)[1:] 
        return counts.tolist()