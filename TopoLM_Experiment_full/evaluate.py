import torch
import os
import pickle
from models.topolm_core import TopoLM

def compute_wasserstein_distance(mu_a, logvar_a, mu_b, logvar_b):
    """计算两个高斯域在拓扑空间中的绝对 2-Wasserstein 距离平方"""
    sigma_a = torch.exp(0.5 * logvar_a)
    sigma_b = torch.exp(0.5 * logvar_b)
    
    dist_mu = torch.sum((mu_a - mu_b) ** 2, dim=-1)
    dist_sigma = torch.sum((sigma_a - sigma_b) ** 2, dim=-1)
    return dist_mu + dist_sigma

def load_topolm_environment(save_dir="checkpoints"):
    print("[加载] 正在唤醒 TopoLM 拓扑大脑...")
    
    # 1. 加载概念词表
    vocab_path = os.path.join(save_dir, "topolm_vocab.pkl")
    with open(vocab_path, "rb") as f:
        word2id = pickle.load(f)
    
    # 2. 实例化模型 (参数需与训练时严格一致)
    VOCAB_SIZE = len(word2id)
    HIDDEN_DIM = 128
    NUM_HEADS = 4
    NUM_LAYERS = 2
    
    model = TopoLM(vocab_size=VOCAB_SIZE, hidden_dim=HIDDEN_DIM, 
                   num_heads=NUM_HEADS, num_layers=NUM_LAYERS)
    
    # 3. 加载拓扑演化权重
    model_path = os.path.join(save_dir, "topolm_v1_weights.pth")
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()
    
    return model, word2id

def evaluate_topological_repulsion(model, word2id, concept_pairs):
    print("\n==================================================")
    print(" 实验 E3：抗幻觉与矛盾互斥测试 (Contradiction Resistance)")
    print("==================================================")
    print(f"{'概念 A':<10} | {'概念 B':<10} | {'空间绝对距离 (W2)':<18} | {'注意力纠缠度 (Attention)'}")
    print("-" * 70)
    
    # 模拟环境常量 (与 topological_attention 中的设定一致)
    tau = 0.5
    domain_dim = 128 // 4 # HIDDEN_DIM // NUM_HEADS
    
    for word_a, word_b in concept_pairs:
        id_a = word2id.get(word_a)
        id_b = word2id.get(word_b)
        
        if id_a is None or id_b is None:
            print(f"⚠️ 单词 {word_a} 或 {word_b} 不在词表中，跳过。")
            continue
            
        # 构建最小输入张量
        context = torch.tensor([[id_a, id_b]], dtype=torch.long)
        
        with torch.no_grad():
            # 前向传播，获取在拓扑空间中的几何参数
            _, mu, logvar = model(context)
            
            mu_a, logvar_a = mu[:, 0, :], logvar[:, 0, :]
            mu_b, logvar_b = mu[:, 1, :], logvar[:, 1, :]
            
            # 计算 Wasserstein 距离
            w2_sq = compute_wasserstein_distance(mu_a, logvar_a, mu_b, logvar_b).item()
            
            # 计算理论上的注意力分数 (拓扑引擎底层计算公式)
            attention_score = torch.exp(torch.tensor(-w2_sq / (tau * (domain_dim ** 0.5)))).item()
            
            # 格式化输出
            if w2_sq > 20.0:
                # 高斥力 (互斥概念)
                dist_str = f"🔴 {w2_sq:>8.2f} (高斥力)"
                attn_str = f"被物理隔离 ({attention_score:.2e})"
            else:
                # 高引力 (包含/相关概念)
                dist_str = f"🟢 {w2_sq:>8.2f} (高引力)"
                attn_str = f"高度纠缠 ({attention_score:.4f})"
                
            print(f"{word_a:<10} | {word_b:<10} | {dist_str:<18} | {attn_str}")

if __name__ == "__main__":
    model, word2id = load_topolm_environment()
    
    # 构建测试集：对比“符合常识”与“违背常识(幻觉)”的概念对
    test_pairs = [
        # 1. 验证层级包含 (E1 领域测试) - 预期高引力，距离近
        ("dog", "animal"),
        ("car", "vehicle"),
        ("apple", "fruit"),
        
        # 2. 验证抗幻觉互斥 (E3 矛盾测试) - 预期高斥力，注意力阻断
        ("fire", "water"),
        ("good", "bad"),
        ("up", "down"),
        ("sleeping", "dancing")
    ]
    
    evaluate_topological_repulsion(model, word2id, test_pairs)