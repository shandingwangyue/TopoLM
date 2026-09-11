"""
TopoLM Cognitive Experiment E4: Compositional Domain Transfer
File: experiments/generalization.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml
from transformers import BertTokenizer
from models.topolm import TopoLM
import torch.nn.functional as F

def run_generalization_experiment():
    with open("configs/base.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg["training"]["device"] if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

    model = TopoLM(
        vocab_size=len(tokenizer),
        num_domains=cfg["model"]["num_domains"],
        d_model=cfg["model"]["d_model"],
        num_heads=cfg["model"]["num_heads"],
        max_seq_len=cfg["model"]["max_seq_len"],
        init_log_sigma=cfg["model"]["init_log_sigma"],
    ).to(device)

    weights_path = "topolm_weights.pt"
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print("Successfully loaded trained TopoLM weights!")
    else:
        print("Warning: Trained weights not found, using baseline.")

    model.eval()

    DOMAIN_PHYSICS = cfg["domains"]["PHYSICS"]
    DOMAIN_FANTASY = cfg["domains"]["FANTASY"]

    # =========================================================================
    # 拓扑代数合成：零样本构建“仙工/机关术 (Xianxia Engineering)”复合新域
    # mu_composite = alpha * mu_physics + (1 - alpha) * mu_fantasy
    # =========================================================================
    alpha = 0.5
    mu_phys = model.topo_space.mu[DOMAIN_PHYSICS]
    mu_fant = model.topo_space.mu[DOMAIN_FANTASY]
    mu_composite = alpha * mu_phys + (1 - alpha) * mu_fant
    # 1. 采用严格高斯方差无偏插值 (消除人工 1.2 膨胀)
    sigma_composite = 0.5 * (model.topo_space.get_sigma()[DOMAIN_PHYSICS] + model.topo_space.get_sigma()[DOMAIN_FANTASY])

    print("=" * 85)
    print("      TopoLM Experiment E4: 跨域拓扑组合与零样本泛化测试 (Compositional Transfer)")
    print("=" * 85)
    print(f"已完成拓扑向量合成：Domain_Composite (50% 物理学 + 50% 仙侠玄幻)")
    print(f"中心位移距离: 到物理域={torch.norm(mu_composite - mu_phys):.2f} | 到仙侠域={torch.norm(mu_composite - mu_fant):.2f}\n")

    test_sequences = [
        {
            "type": "纯物理文本",
            "text": "热力学系统在没有能量补充时无法对外持续做功。",
        },
        {
            "type": "纯仙侠文本",
            "text": "风雪客运转浑厚真气，凌空踏步，直冲九重天阙。",
        },
        {
            "type": "跨界复合命题 (仙工/机关术/玄幻科技)",
            "text": "机关傀儡以灵石阵法转化热力做功，借助机械飞轮驱动悬浮法器。",
        }
    ]

    for item in test_sequences:
        tokens = tokenizer(item["text"], return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
        with torch.no_grad():
            emb = model.token_emb(tokens)
            
            # 计算到原有两个域的马氏势能
            e_p = model.topo_space.mahalanobis_sq(emb, DOMAIN_PHYSICS).mean().item()
            e_f = model.topo_space.mahalanobis_sq(emb, DOMAIN_FANTASY).mean().item()
            
            # 计算到零样本合成域的马氏势能
            diff_comp = emb - mu_composite
            e_comp = (((diff_comp ** 2) / sigma_composite).sum(dim=-1)).mean().item()

        print(f"[测试样本类型]: {item['type']}")
        print(f"输入文本: \"{item['text']}\"")
        print(f"  -> 物理域势能 E(x, Physics):   {e_p:.2f}")
        print(f"  -> 仙侠域势能 E(x, Fantasy):   {e_f:.2f}")
        print(f"  -> 合成域势能 E(x, Composite): {e_comp:.2f}")

        # 检查合成域是否显著包容了复合文本
        if e_comp < e_p and e_comp < e_f:
            verdict = "OPTIMAL FIT in Composite Domain (成功以更低势能捕获复合跨域概念)"
        else:
            verdict = "ALIGNED to Single Domain"
        print(f"  -> 拓扑泛化裁决: {verdict}\n")
    energies = torch.tensor([e_p, e_f, e_comp])
    # 使用负半势能转换为 Logits
    logits = -0.5 * (energies - energies.min()) # 防溢出
    probs = F.softmax(logits, dim=0) * 100

    print(f"[测试样本类型]: {item['type']}")
    print(f"输入文本: \"{item['text']}\"")
    print(f"  -> 势能分布: 物理={e_p:.2f} | 仙侠={e_f:.2f} | 合成={e_comp:.2f}")
    print(f"  -> 归属置信度: 物理={probs[0]:.2f}% | 仙侠={probs[1]:.2f}% | 合成域={probs[2]:.2f}%")

    pred_domain = torch.argmax(probs).item()
    domain_labels = ["Physics", "Fantasy", "Composite (仙工复合)"]
    print(f"  -> 最优拓扑归属: [{domain_labels[pred_domain]}]\n")
    print("=" * 85)


if __name__ == "__main__":
    run_generalization_experiment()