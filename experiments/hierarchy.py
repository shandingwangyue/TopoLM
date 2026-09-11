"""
TopoLM Cognitive Experiment E2: Hierarchical Transitivity
File: experiments/hierarchy.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from models.domain_space import TopologicalDomainSpace


def run_hierarchy_experiment():
    d_model = 64
    # 定义包含关系的 4 个测试域：0: Reality, 1: Physics, 2: Gravity, 3: Fantasy (独立异域)
    domain_space = TopologicalDomainSpace(num_domains=4, d_model=d_model, init_log_sigma=0.5)

    with torch.no_grad():
        # 手动构造层级包络几何：
        # Reality: 中心位于原点，极大体积
        domain_space.mu[0].zero_()
        domain_space.log_sigma[0].fill_(2.5)  # 半径极大

        # Physics: 稍有偏移，但落入 Reality 内，中等体积
        domain_space.mu[1] = domain_space.mu[0] + 0.3
        domain_space.log_sigma[1].fill_(1.2)

        # Gravity: 进一步细分叶子，落入 Physics 内，小体积
        domain_space.mu[2] = domain_space.mu[1] + 0.2
        domain_space.log_sigma[2].fill_(0.4)

        # Fantasy: 远离 Reality 中心
        domain_space.mu[3].fill_(10.0)
        domain_space.log_sigma[3].fill_(1.2)

    print("=" * 85)
    print("        TopoLM Experiment E2: 层级包含与偏序传递性测试 (Hierarchy Probe)")
    print("=" * 85)

    test_chains = [
        (2, 1, "Gravity ⊂ Physics (直接子域关系)"),
        (1, 0, "Physics ⊂ Reality (直接子域关系)"),
        (2, 0, "Gravity ⊂ Reality (层级传递性: A ⊂ B 且 B ⊂ C => A ⊂ C)"),
        (3, 0, "Fantasy ⊂ Reality (互斥域非包含关系检验)"),
    ]

    for sub_idx, sup_idx, relation_name in test_chains:
        margin = domain_space.containment_margin(sub_idx, sup_idx).item()
        center_dist = domain_space.domain_centroid_distance(sub_idx, sup_idx).item()
        # 裕度 > 0 说明子域的高斯包络完全被父域覆盖
        is_contained = margin > 0.0
        status = "PASSED (Valid Containment)" if is_contained else "REJECTED (Disjoint / Breach)"

        print(f"\n关系测试: {relation_name}")
        print(f"  -> 中心几何间距: {center_dist:.4f}")
        print(f"  -> 拓扑覆盖裕度 (R_sup - R_sub - Dist): {margin:.4f}")
        print(f"  -> 偏序判定结果: [{status}]")

    print("\n" + "=" * 85)


if __name__ == "__main__":
    run_hierarchy_experiment()