"""
TopoLM Cognitive Experiment E3: Contradiction Resistance
File: experiments/contradiction.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml
from transformers import BertTokenizer
from models.topolm import TopoLM


def run_contradiction_experiment():
    with open("configs/base.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg["training"]["device"] if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

    # 加载已训练完成的模型骨架 (或测试实例)
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
    # 设为评估模式
    model.eval()

    DOMAIN_PHYSICS = cfg["domains"]["PHYSICS"]
    DOMAIN_FANTASY = cfg["domains"]["FANTASY"]

    print("=" * 85)
    print("        TopoLM Experiment E3: 认知矛盾与拓扑势能摩擦测试 (Contradiction Probe)")
    print("=" * 85)

    test_propositions = [
        {
            "category": "物理真实常识 (Valid Physics)",
            "text": "孤立系统的熵永远不会自发减少，重物受引力落向地面。",
            "expected_domain": "Physics",
        },
        {
            "category": "物理悖论命题 (Physical Violation / Paradox)",
            "text": "一个没有任何支撑的机械装置在真空中持续无能量静止悬浮。",
            "expected_domain": "Violation / Rejection",
        },
        {
            "category": "仙侠流派叙事 (Valid Fantasy Context)",
            "text": "十万里风雪客运转真气，踏破虚空直上九霄云层。",
            "expected_domain": "Fantasy",
        },
    ]

    for item in test_propositions:
        tokens = tokenizer(item["text"], return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
        with torch.no_grad():
            emb = model.token_emb(tokens)
            # 计算马氏距离平方 (拓扑势能能量 E)
            energy_phys = model.topo_space.mahalanobis_sq(emb, DOMAIN_PHYSICS).mean().item()
            energy_fant = model.topo_space.mahalanobis_sq(emb, DOMAIN_FANTASY).mean().item()

        # 归一化为相对置信度
        p_p = torch.exp(torch.tensor(-0.5 * min(energy_phys, 50.0)))
        p_f = torch.exp(torch.tensor(-0.5 * min(energy_fant, 50.0)))
        conf_phys = (p_p / (p_p + p_f + 1e-9)).item() * 100
        conf_fant = (p_f / (p_p + p_f + 1e-9)).item() * 100

        print(f"\n[命题类型]: {item['category']}")
        print(f"输入文本: \"{item['text']}\"")
        print(f"  -> 物理域违反势能 E(x, Physics): {energy_phys:.2f} | 归属置信度: {conf_phys:.2f}%")
        print(f"  -> 仙侠域违反势能 E(x, Fantasy): {energy_fant:.2f} | 归属置信度: {conf_fant:.2f}%")

        # 设定物理域常态认知接纳阈值 (正常物理句子势能在 46 左右，设定 52.0 为违规红线)
        ENERGY_THRESHOLD = 52.0

        print(f"\n[命题类型]: {item['category']}")
        print(f"输入文本: \"{item['text']}\"")
        print(f"  -> 物理域违反势能 E(x, Physics): {energy_phys:.2f} | 归属置信度: {conf_phys:.2f}%")
        print(f"  -> 仙侠域违反势能 E(x, Fantasy): {energy_fant:.2f} | 归属置信度: {conf_fant:.2f}%")

        # 具备物理常识防御的双重认知裁决逻辑
        if energy_phys > ENERGY_THRESHOLD and energy_fant > ENERGY_THRESHOLD:
            verdict = "CRITICAL VIOLATION: 物理法则严重违规 (双域势能激增，判定为伪科学悖论/非法命题)"
        elif energy_phys <= energy_fant and energy_phys <= ENERGY_THRESHOLD:
            verdict = "ACCEPTED by Physics Domain (客观物理真理，符合经典力学/热力学)"
        elif energy_fant < energy_phys and energy_fant <= ENERGY_THRESHOLD:
            verdict = "ACCEPTED by Fantasy Domain (属于文学虚构/仙侠语境，免除物理因果惩罚)"
        else:
            verdict = "REJECTED by Physics: 超出物理公理边界，建议分流至虚构认知域"

        print(f"  -> 认知架构自发裁决: {verdict}")

    print("\n" + "=" * 85)


if __name__ == "__main__":
    run_contradiction_experiment()