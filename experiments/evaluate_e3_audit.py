"""
TopoLM: Section 9.3 E3 Integrity Audit
复现论文 Table 3：验证模型是否学会了包含与绝对排斥
"""
import os, sys, argparse, torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.topolm import TopoLM

def audit_concept_pairs(model, word2id, pairs):
    print(f"{'Concept A':<15} {'Concept B':<15} {'Dist(uA,uB)':>12} {'Contain Margin':>15} {'Status':>20}")
    print("-" * 80)
    for word_a, word_b in pairs:
        if word_a not in word2id or word_b not in word2id: 
            continue
        id_a, id_b = word2id[word_a], word2id[word_b]
        mu_a, r_a = model.topo_space.mu[id_a], model.topo_space.get_radius(id_a)
        mu_b, r_b = model.topo_space.mu[id_b], model.topo_space.get_radius(id_b)
        
        dist = torch.norm(mu_a - mu_b, p=2).item()
        containment = (r_b - r_a - dist).item() 
        status = "Strict Contain" if containment > 0 else "Disjoint/Intersect"
        if (r_a + r_b + 2.0 - dist).item() < 0: 
            status = "Absolute Repulsion"
        print(f"{word_a:<15} {word_b:<15} {dist:12.4f} {containment:15.4f} {status:>20}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase1-ckpt", type=str, default="checkpoints/topolm_phase1_emergence.pt")
    args = parser.parse_args()

    print("=== TopoLM E3 Integrity Audit (Real Checkpoint) ===")
    
    try:
        from data.topological_dataset import RealConceptVocab, WordNetTopologyParser
        vocab = RealConceptVocab()
        WordNetTopologyParser.extract_relations(vocab)
        word2id = vocab.word2id
        vocab_size = len(vocab)
        
        if os.path.exists(args.phase1_ckpt):
            ckpt_state = torch.load(args.phase1_ckpt, map_location="cpu")
            num_domains = ckpt_state["mu"].shape[0]
            
            # 使用真实的 128 维度和真实 domain 数量实例化
            model = TopoLM(vocab_size=max(vocab_size, num_domains), num_domains=num_domains, d_model=128)
            model.topo_space.load_state_dict(ckpt_state)
            print(f"[*] 成功加载真实 6万级 物理骨架: {args.phase1_ckpt}\n")
            
            # 测试论文里的核心概念对 (Table 3)
            test_pairs = [
                ("husky", "dog"),
                ("dog", "animal"),
                ("animal", "entity"),
                ("fire", "water"),
                ("hot", "cold"),
                ("good", "bad")
            ]
            audit_concept_pairs(model, word2id, test_pairs)
        else:
            print(f"[!] 找不到权重文件 {args.phase1_ckpt}，请先运行 run_phase1_emergence.py")
            
    except Exception as e:
        print(f"[!] 运行失败: {e}")