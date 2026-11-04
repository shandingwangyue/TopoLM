import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pickle
import os
import math
import nltk

# 首次运行需要下载语料
# nltk.download('brown')
from nltk.corpus import brown

# =====================================================================
# 1. 纯净语言数据集 (只保留已建立物理坐标的概念)
# =====================================================================
class TopologicalLanguageDataset(Dataset):
    def __init__(self, vocab, seq_len=32):
        print("[数据] 正在加载并过滤无监督语言语料 (Brown Corpus)...")
        raw_words = brown.words()
        
        self.seq_len = seq_len
        self.data = []
        
        # 将文本转化为物理宇宙中的坐标 ID
        # 如果语料库里的词在我们的 6万 物理词表中没有，直接跳过（保持常识宇宙的纯净）
        current_seq = []
        for w in raw_words:
            w_lower = w.lower()
            if w_lower in vocab.word2id:
                current_seq.append(vocab.word2id[w_lower])
                if len(current_seq) == seq_len + 1:  # +1 是为了错位预测 Next-Token
                    self.data.append(current_seq)
                    current_seq = []
                    
        print(f"[数据] 成功构建 {len(self.data)} 条有效常识上下文序列。")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = torch.tensor(self.data[idx], dtype=torch.long)
        return seq[:-1], seq[1:] # Input, Target

# =====================================================================
# 2. 神经符号混合架构 (Neuro-Symbolic Transformer)
# =====================================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0)]

class TopoLM(nn.Module):
    def __init__(self, vocab_size, dim=128, nhead=8, num_layers=4):
        super().__init__()
        self.dim = dim
        
        # ----------------------------------------------------
        # 【核心物理基座】代替传统的随机 Embedding
        # ----------------------------------------------------
        self.mu = nn.Parameter(torch.zeros(vocab_size, dim))
        self.log_radius = nn.Parameter(torch.zeros(vocab_size, 1))
        
        # ----------------------------------------------------
        # 【语言血肉网络】Transformer 层
        # ----------------------------------------------------
        self.pos_encoder = PositionalEncoding(dim)
        encoder_layers = nn.TransformerEncoderLayer(d_model=dim, nhead=nhead, dim_feedforward=dim*4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # 语言输出头 (将 128 维空间坐标重新投影回概率词表)
        self.lm_head = nn.Linear(dim, vocab_size)

    def load_and_freeze_physical_universe(self, ckpt_path):
        """极其关键的步骤：加载第一性原理，并实施物理封锁"""
        print(f"\n[*] 正在加载 Phase 1 常识拓扑骨架: {ckpt_path}")
        state_dict = torch.load(ckpt_path)
        
        # 强制注入绝对坐标
        self.mu.data = state_dict['mu']
        self.log_radius.data = state_dict['log_radius']
        
        # ============================================================
        # 🌟 拓扑边界冻结法则 (Zero-Hallucination Protocol) 🌟
        # 切断梯度，任何语言语料都无法篡改概念在宇宙中的绝对体积与位置！
        # ============================================================
        self.log_radius.requires_grad = False
        self.mu.requires_grad = False 
        
        print("[✔] 物理封锁已激活: 宇宙骨架 (mu, log_radius) 梯度已切断！只允许 Transformer 生长。")

    def forward(self, src):
        # 1. 实体化：直接使用物理宇宙的绝对中心点作为语言特征的起点
        x = self.mu[src] * math.sqrt(self.dim)
        
        # 2. 注入语法时序
        x = self.pos_encoder(x)
        
        # 3. 语言注意力纠缠 (因果 Mask，防止偷看未来)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(src.size(1)).to(src.device)
        output = self.transformer(x, mask=causal_mask, is_causal=True)
        
        # 4. 坍缩为概率分布
        logits = self.lm_head(output)
        return logits

# =====================================================================
# 3. 语言血肉演化循环
# =====================================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] 启动 Phase 2 语言注模系统 | 硬件: {device}")
    
    # 1. 唤醒 Phase 1 的常识网格
    # 请确保路径指向你 Phase 1 保存的位置
    vocab_path = "./checkpoints/topolm_emergence_vocab.pkl"
    ckpt_path = "./checkpoints/topolm_phase1_emergence.pth"
    
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"找不到词表: {vocab_path}，请确认路径！")
        
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)
    print(f"[*] 成功唤醒词表矩阵，总概念数: {len(vocab)}")
    
    # 2. 构建纯净语料装载器
    dataset = TopologicalLanguageDataset(vocab, seq_len=32)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
    
    # 3. 实例化 TopoLM 并注入冻结骨架
    model = TopoLM(vocab_size=len(vocab), dim=128, nhead=8, num_layers=4).to(device)
    model.load_and_freeze_physical_universe(ckpt_path)
    
    # 4. 优化器 (注意：这里只会优化 Transformer 和 lm_head，物理坐标毫发无损！)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0005)
    
    print("\n==================================================")
    print(" 开始无监督血肉填充 (语言模型的交叉熵无法污染常识)")
    print("==================================================")
    
    EPOCHS = 30
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            logits = model(inputs)
            
            # (Batch*Seq, Vocab)
            loss = criterion(logits.view(-1, len(vocab)), targets.view(-1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 50 == 0 and batch_idx > 0:
                print(f"  ↳ Epoch {epoch} | Batch {batch_idx:03d} | LM Loss: {loss.item():.4f}")
                
        print(f"[!] Epoch {epoch} 结束 | 平均语言损失: {total_loss/len(dataloader):.4f}")

    print("\n[✔] Phase 2 语言注模完成！")
    
    # 5. 保存具备了常识与语言双重能力的最终 AGI 雏形
    os.makedirs("checkpoints", exist_ok=True)
    final_model_path = "checkpoints/topolm_phase2_final.pth"
    torch.save(model.state_dict(), final_model_path)
    print(f"[+] 终极认知大脑已保存至: {final_model_path}")

if __name__ == "__main__":
    main()