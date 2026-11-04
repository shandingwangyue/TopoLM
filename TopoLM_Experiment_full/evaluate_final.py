import torch
import torch.nn.functional as F
import pickle
import math
import torch.nn as nn

# ==========================================
# 重构极其精简的模型外壳，仅用于探测物理状态
# ==========================================
class TopoLM_Probe(nn.Module):
    def __init__(self, vocab_size, dim=128):
        super().__init__()
        self.mu = nn.Parameter(torch.zeros(vocab_size, dim))
        self.log_radius = nn.Parameter(torch.zeros(vocab_size, 1))

    def get_radius(self, ids):
        return (F.softplus(self.log_radius[ids]) + 0.1).squeeze().item()

    def get_distance(self, id_A, id_B):
        dist = torch.norm(self.mu[id_A] - self.mu[id_B], dim=-1)
        return dist.item()

def main():
    print("[*] 正在唤醒 TopoLM 终极认知大脑 (Phase 2 Final) ...")
    vocab_path = "./checkpoints/topolm_emergence_vocab.pkl"
    ckpt_path = "checkpoints/topolm_phase2_final.pth"
    
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)
        
    model = TopoLM_Probe(vocab_size=len(vocab), dim=128)
    
    # 加载最终模型 (允许 strict=False 因为探测器不包含 Transformer 层)
    state_dict = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    
    print("\n==================================================================")
    print(" 终极 E3 实验：大模型语言冲刷后的常识拓扑状态 (Zero-Hallucination)")
    print("==================================================================")
    print(f"{'概念 A':<10} | {'概念 B':<12} | {'A 的体积(σ)':<12} | {'B 的体积(σ)':<12} | {'绝对空间距离'}")
    print("-" * 66)
    
    # 黄金测试集
    test_pairs = [
        # 1. 结构嵌套验证 (A 必须包含于 B)
        ("dog", "animal"),
        ("husky", "dog"),
        ("leaf", "tree"),
        ("cat", "animal"),
        ("tiger", "animal"),
        
        # 2. 绝对排斥验证 (必须被物理隔离)
        ("monkey","tree"),
        ("fire", "water"),
        ("good", "bad"),
        ("up", "down")
    ]
    
    for word_A, word_B in test_pairs:
        if word_A not in vocab.word2id or word_B not in vocab.word2id:
            continue
            
        id_A = vocab.word2id[word_A]
        id_B = vocab.word2id[word_B]
        
        radius_A = model.get_radius(id_A)
        radius_B = model.get_radius(id_B)
        distance = model.get_distance(id_A, id_B)
        
        print(f"{word_A:<10} | {word_B:<12} | {radius_A:<12.4f} | {radius_B:<12.4f} | {distance:.4f}")

if __name__ == "__main__":
    main()