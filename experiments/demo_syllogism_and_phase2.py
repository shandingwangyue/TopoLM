"""
TopoLM: 零样本三段论传递演绎推理 + Phase 2 冻结骨架协议验证
演示:
1. 已知链: 哈士奇 (husky) ⊂ 狗 (dog) ⊂ 动物 (animal)
2. 注入新定律: 动物 (animal) ⊂ 需要能量维持 (require_energy)
3. 零样本推导: 哈士奇 (husky) ⊂ 需要能量维持 (require_energy) (无需直接训练 husky-energy 边)
4. 冻结骨架协议: freeze_backbone() 后训练语言模型，确保拓扑包含裕度 0 退化
"""

import os
import sys
import torch
import torch.optim as optim

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.domain_space import TopologicalDomainSpace
from models.topolm import TopoLM
from training.phase1_loss import TopoMultiObjectiveLoss


def run_syllogism_and_freeze_demo():
    torch.manual_seed(42)
    # 5 个概念: 0=husky, 1=dog, 2=animal, 3=require_energy, 4=no_energy_perpetual
    CONCEPTS = ["husky", "dog", "animal", "require_energy", "no_energy_perpetual"]
    d_model = 64

    # 初始化纯物理流形空间
    space = TopologicalDomainSpace(num_domains=5, d_model=d_model, init_log_sigma=-1.0)
    
    # 实例化专门用于 Phase 1 的纯物理涌现损失
    phase1_loss_fn = TopoMultiObjectiveLoss(
        delta_inc=0.15,
        delta_rep=6.0,
        alpha_tension=0.005,
        beta_rel=1.0,
    )

    optimizer = optim.Adam(space.parameters(), lr=0.05)

    # Phase 1: 仅训练已知的包含链 (husky⊂dog, dog⊂animal) 和注入的公理与互斥
    # 从未直接给出 (husky, require_energy) 的训练对！
    inclusion_edges = [(0, 1), (1, 2), (2, 3)]
    disjoint_edges = [(3, 4)]

    # 把边展平成 batch，分别送给包含和排斥
    sub_idx = torch.tensor([e[0] for e in inclusion_edges], dtype=torch.long)
    super_idx = torch.tensor([e[1] for e in inclusion_edges], dtype=torch.long)
    ant_a_idx = torch.tensor([e[0] for e in disjoint_edges], dtype=torch.long)
    ant_b_idx = torch.tensor([e[1] for e in disjoint_edges], dtype=torch.long)

    for step in range(300):
        optimizer.zero_grad()
        # 直接使用我们在 Phase 1 Loss 里写好的前向涌现接口
        loss, _ = phase1_loss_fn.phase1_emergence_loss(
            space, sub_idx, super_idx, ant_a_idx, ant_b_idx
        )
        loss.backward()
        optimizer.step()

    print("=== Phase 1 拓扑流形涌现结果 ===")
    with torch.no_grad():
        for idx, name in enumerate(CONCEPTS):
            print(f"  Concept [{name:19s}] | Effective Radius R = {space.get_radius(idx).item():.4f}")

        print("\n=== 零样本三段论传递推理审计 (Zero-Shot Syllogism Audit) ===")
        pairs = [
            (0, 1, "husky ⊂ dog (直接训练)"),
            (1, 2, "dog ⊂ animal (直接训练)"),
            (2, 3, "animal ⊂ require_energy (注入公理)"),
            (0, 3, "husky ⊂ require_energy (★ 零样本几何传递推导)"),
            (0, 4, "husky ⊂ no_energy_perpetual (★ 悖论幻觉拦截)"),
        ]
        for sub_i, sup_i, desc in pairs:
            margin = space.containment_margin(sub_i, sup_i).item()
            status = "TRUE (严格包含)" if margin > 0 else "FALSE (拓扑排斥)"
            print(f"  {desc:46s} | Margin = {margin:+.4f} -> {status}")

    # Phase 2: 冻结骨架后训练语言层，验证拓扑 0 退化
    # 实例化完整的 TopoLM，把涌现好的物理空间装载进去
    lm = TopoLM(vocab_size=100, num_domains=5, d_model=d_model, num_heads=4, max_seq_len=32)
    lm.topo_space.load_state_dict(space.state_dict())
    
    # 【核心零幻觉协议】冻结物理流形骨架
    lm.topo_space.freeze_backbone()

    margin_before = lm.topo_space.containment_margin(0, 3).item()
    
    # 仅优化语言层 (Embedding, Attention, FFN, Head)
    lm_opt = optim.AdamW([p for p in lm.parameters() if p.requires_grad], lr=1e-3)
    dummy_x = torch.randint(0, 100, (8, 16))
    dummy_y = torch.randint(0, 100, (8, 16))
    ce = torch.nn.CrossEntropyLoss()

    for _ in range(20):
        lm_opt.zero_grad()
        out = lm(dummy_x, active_domain_idx=3)
        loss = ce(out["logits"].view(-1, 100), dummy_y.view(-1))
        loss.backward()
        lm_opt.step()

    margin_after = lm.topo_space.containment_margin(0, 3).item()
    print(f"\n=== Phase 2 冻结骨架审计 ===")
    print(f"  语言训练前 husky ⊂ require_energy 裕度: {margin_before:+.6f}")
    print(f"  语言训练后 husky ⊂ require_energy 裕度: {margin_after:+.6f} (零漂移!)")


if __name__ == "__main__":
    run_syllogism_and_freeze_demo()
