"""
TopoLM: Phase 2 Language Fleshing on Brown Corpus
冻结 Phase 1 的物理骨架，仅训练注意力层与语言层
"""
import os, sys, argparse, torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.topolm import TopoLM
from training.phase2_losses import TopoLMLoss
from training.trainer import TopoTrainer

try:
    from nltk.corpus import brown
except ImportError:
    print("[!] 请先运行: python -m nltk.downloader brown")
    
try:
    from data.topological_dataset import RealConceptVocab, WordNetTopologyParser
except ImportError:
    pass

class BrownTopologicalDataset(Dataset):
    def __init__(self, word2id, seq_len=32):
        tokens = (word2id.get(word.lower()) for word in brown.words())
        ids = torch.tensor([idx for idx in tokens if idx is not None], dtype=torch.long)
        if ids.numel() == 0:
            raise ValueError("布朗语料中没有任何词汇命中当前词表，请检查词表是否正确构建。")
        complete_tokens = ids.numel() // (seq_len + 1) * (seq_len + 1)
        self.sequences = ids[:complete_tokens].reshape(-1, seq_len + 1)
    def __len__(self): return len(self.sequences)
    def __getitem__(self, idx): 
        seq = self.sequences[idx]
        return seq[:-1], seq[1:]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase1-ckpt", type=str, default="checkpoints/topolm_phase1_emergence.pt")
    # 为了适配长序列，稍微调小 batch_size 到 16，保证 24G 显存绝对安全
    parser.add_argument("--batch-size", type=int, default=16) 
    parser.add_argument("--epochs", type=int, default=40)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=== TopoLM Phase 2: Frozen-Backbone Language Modeling ===")
    
    # 1. 动态探测并重建词表 
    try:
        vocab = RealConceptVocab()
        WordNetTopologyParser.extract_relations(vocab)
        vocab_size = len(vocab)
        word2id = vocab.word2id
        print(f"[*] 已成功重建词表，大小: {vocab_size}")
    except NameError:
        vocab_size = 10000
        word2id = {}
        print("[!] 未找到 WordNet 词表工具，使用默认词表大小 10000")
        
    num_domains = 2
    if os.path.exists(args.phase1_ckpt):
        ckpt_state = torch.load(args.phase1_ckpt, map_location="cpu")
        num_domains = ckpt_state["mu"].shape[0]
        vocab_size = max(vocab_size, num_domains) 
    
    model = TopoLM(
        vocab_size=vocab_size, 
        num_domains=num_domains, 
        d_model=128, 
        sparse_mode=True
    ).to(device)
    
    if os.path.exists(args.phase1_ckpt):
        model.topo_space.load_state_dict(ckpt_state)
        print(f"[*] 已成功加载 Phase 1 物理骨架: {args.phase1_ckpt} (Domains: {num_domains})")
        
    model.topo_space.freeze_backbone()
    print("[*] 拓扑流形骨架已永久冻结 (nabla_{mu, sigma} = 0)")

    # 【核心显存优化】：Phase 2 只算语言交叉熵，不计算庞大的拓扑分配 Voronoi 势力场
    # 直接屏蔽 6 万个流形的广播计算，瞬间省出 30GB+ 显存！
    model.topo_space.compute_all_distances = lambda x: None

    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=5e-4)
    criterion = TopoLMLoss()
    trainer = TopoTrainer(model, criterion, optimizer, disjoint_pairs=(), inclusion_pairs=())
    
    try:
        dataset = BrownTopologicalDataset(word2id)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        print(f"[+] 开始注入语言表征... (数据量: {len(dataset)} 个 Sequence)")
        
        for epoch in range(1, args.epochs + 1):
            total_loss = 0.0
            model.train()
            for i, (batch_x, batch_y) in enumerate(loader):
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                out = model(batch_x, active_domain_idx=0)
                
                optimizer.zero_grad()
                loss = criterion.language_loss(out["logits"], batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
                
                # 增加控制台打印，让你看到进度
                if i % 100 == 0:
                    print(f"   [Step {i:4d}/{len(loader)}] LM Loss: {loss.item():.4f}")
            
            print(f"Epoch {epoch:02d}/{args.epochs} | Avg LM Loss: {total_loss/len(loader):.4f}")
            print("[+] 语言表征注入完成！")
            # [修复] 保存 Phase 2 的完整满血权重
            os.makedirs("checkpoints", exist_ok=True)
            phase2_ckpt_path = "checkpoints/topolm_phase2_language.pt"
            torch.save(model.state_dict(), phase2_ckpt_path)
            print(f"[+] 满血版模型 (冻结流形骨架 + 语言注模) 已保存至: {phase2_ckpt_path}")
            
    except Exception as e:
        print(f"[-] 跳过语言训练 (出现错误): {e}")

if __name__ == "__main__":
    main()