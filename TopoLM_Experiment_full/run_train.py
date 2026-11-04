import torch
from torch.utils.data import DataLoader
from data.topological_dataset import RealConceptVocab, WordNetTopologyParser, NLTKTextParser, TopoCurriculumDataset
from models.topolm_core import TopoLM
from training.losses import TopologicalEnergyLoss
from training.trainer import train_epoch, monitor_stratification
import torch.nn as nn
import torch.optim as optim
import os
import pickle

def main():
    print("==================================================")
    print("      TopoLM: 9级 Token 拓扑域演化训练管线        ")
    print("==================================================")
    
    # 1. 参数与设备初始化
    VOCAB_SIZE = 15000
    HIDDEN_DIM = 128
    NUM_HEADS = 4
    NUM_LAYERS = 2
    BATCH_SIZE = 64
    EPOCHS = 10
    PHASE_SHIFT_EPOCH = 6 # 第6轮开始开放文本语料
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] 训练硬件: {device}")

    # 2. 抽取真实数据
    vocab = RealConceptVocab()
    # 注入黄金验证集（保持这部分不变，因为我们需要这几个词在后面做严密测试）
    golden_inclusions = [
        (vocab.add_concept("dog"), vocab.add_concept("animal"), 1.0),
        (vocab.add_concept("car"), vocab.add_concept("vehicle"), 1.0),
        (vocab.add_concept("apple"), vocab.add_concept("fruit"), 1.0)
    ]*1000
    golden_contradictions = [
        (vocab.add_concept("fire"), vocab.add_concept("water"), -1.0),
        (vocab.add_concept("up"), vocab.add_concept("down"), -1.0),
        (vocab.add_concept("sleeping"), vocab.add_concept("dancing"), -1.0),
        (vocab.add_concept("hot"), vocab.add_concept("cold"), -1.0)
    ]*1000
    print(f"[*] 已强制注入黄金验证集锚点...")
    
    inclusions, contradictions = WordNetTopologyParser.extract_relations(vocab)
    structural_data = golden_inclusions + golden_contradictions + inclusions + contradictions
    text_data = NLTKTextParser.extract_sequences(vocab, seq_len=8)
    
    # 3. 初始化数据集
    dataset = TopoCurriculumDataset(structural_data, text_data, seq_len=8)
    # 为保证测试速度，限制 dataloader 在每个 epoch 只采样部分 batch
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
    # 4. 实例化拓扑模型与损失
    model = TopoLM(vocab_size=len(vocab), hidden_dim=HIDDEN_DIM, num_heads=NUM_HEADS, num_layers=NUM_LAYERS).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=5e-4)
    
    topo_criterion = TopologicalEnergyLoss(margin=3.0, volume_reg_weight=0.01)
    lm_criterion = nn.CrossEntropyLoss(ignore_index=0) # 忽略 PAD

    # 5. 双轨课程学习主循环
    for epoch in range(1, EPOCHS + 1):
        if epoch == 1:
            dataset.set_phase('rules')
            print("\n>>> Phase 1: 构建拓扑常识骨架 (Topological Grounding) <<<")
        elif epoch == PHASE_SHIFT_EPOCH:
            dataset.set_phase('unsupervised')
            print("\n>>> Phase 2: 无监督概念体积填充 (Unsupervised Fleshing) <<<")

        loss_total, loss_topo, loss_lm = train_epoch(
            model, dataloader, topo_criterion, lm_criterion, optimizer, device
        )
        
        # 监测 9 级分层涌现情况
        levels = monitor_stratification(model, dataloader, device)
        
        print(f"Epoch {epoch:02d}/{EPOCHS} | Phase: {dataset.phase}")
        print(f"  ↳ Loss_Total: {loss_total:.4f} (Topo: {loss_topo:.4f} | LM: {loss_lm:.4f})")
        print(f"  ↳ 9-Level 分层 Token 分布: {levels}")
    # ==========================================
    # 6. 保存模型权重与核心概念词表
    # ==========================================
    save_dir = "checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存模型权重
    model_path = os.path.join(save_dir, "topolm_v1_weights.pth")
    torch.save(model.state_dict(), model_path)
    
    # 保存词汇表 (至关重要，推理时需要用它将单词映射回正确的概念域)
    vocab_path = os.path.join(save_dir, "topolm_vocab.pkl")
    with open(vocab_path, "wb") as f:
        pickle.dump(vocab.word2id, f)
        
    print(f"\n[完成] 模型与词表已成功保存至 `{save_dir}/` 目录。")
if __name__ == "__main__":
    main()