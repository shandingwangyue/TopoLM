"""
TopoLM: Phase 1 Physical Ontology Emergence (WordNet Real Data)
File: experiments/run_phase1_emergence.py
"""
import os, sys, argparse, torch
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.domain_space import TopologicalDomainSpace
from training.phase1_loss import TopoMultiObjectiveLoss

try:
    from data.topological_dataset import RealConceptVocab, TopoCurriculumDataset, WordNetTopologyParser
except ImportError:
    pass

def export_and_plot_academic_figure(radii, id2word, save_dir):
    print(f"\n[*] 正在生成学术级实证图表与 CSV 数据...")
    os.makedirs(save_dir, exist_ok=True)
    r_numpy = radii.squeeze().numpy()
    data = [{"Concept": id2word.get(idx, f"UNK_{idx}"), "Radius": r} for idx, r in enumerate(r_numpy)]
    df = pd.DataFrame(data).sort_values(by="Radius", ascending=False).reset_index(drop=True)
    csv_path = os.path.join(save_dir, "topolm_emergence_radii.csv")
    df.to_csv(csv_path, index=False)
    print(f"[+] 概念体积数据已持久化至: {csv_path}")

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    ax.hist(df['Radius'], bins=100, color='#2c3e50', alpha=0.85, log=True, edgecolor='white')
    ax.set_title("TopoLM Concept Radius Distribution", fontsize=15, fontweight='bold', pad=15)
    ax.set_xlabel("Concept Radius", fontsize=13)
    ax.set_ylabel("Number of Concepts (Log Scale)", fontsize=13)
    
    top_concepts = df.head(5)
    for i, row in enumerate(top_concepts.itertuples(index=False)):
        ax.axvline(x=row.Radius, color="#e74c3c", linestyle="--", alpha=0.7)
        ax.annotate(row.Concept, xy=(row.Radius, 0.9 - 0.1 * i),
                    xycoords=ax.get_xaxis_transform(), xytext=(-8, 0),
                    textcoords="offset points", ha="right", color="#c0392b")

    plt.tight_layout()
    plot_path = os.path.join(save_dir, "emergence_distribution.png")
    plt.savefig(plot_path, format='png', bbox_inches='tight')
    plt.close()

def train_phase1_epoch(manifold, loader, optimizer, loss_fn, device, vocab_size):
    manifold.train()
    total_loss_accum = 0.0
    for batch in loader:
        optimizer.zero_grad()
        edges = batch[0] if isinstance(batch, (list, tuple)) else batch
        edges = edges.to(device)
        
        # Safely extract sub/super idx, filtering out out-of-bounds indices
        # caused by vocabulary mismatch in WordNet
        sub_idx = edges[:, 0]
        super_idx = edges[:, 1]
        valid_mask = (sub_idx < vocab_size) & (super_idx < vocab_size) & (sub_idx >= 0) & (super_idx >= 0)
        
        sub_idx = sub_idx[valid_mask]
        super_idx = super_idx[valid_mask]

        if len(sub_idx) == 0:
            continue

        ant_a_idx = torch.tensor([], dtype=torch.long, device=device)
        ant_b_idx = torch.tensor([], dtype=torch.long, device=device)
        
        loss, _ = loss_fn.phase1_emergence_loss(manifold, sub_idx, super_idx, ant_a_idx, ant_b_idx)
        loss.backward()
        optimizer.step()
        total_loss_accum += loss.item()
    return total_loss_accum / max(1, len(loader))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--output-dir", type=str, default="./checkpoints")
    parser.add_argument("--results-dir", type=str, default="./results")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Starting TopoLM Phase 1 Emergence on {device} ===")

    try:
        vocab = RealConceptVocab()
        inclusions, contradictions = WordNetTopologyParser.extract_relations(vocab)
        dataset = TopoCurriculumDataset(inclusions + contradictions, [], seq_len=2)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        vocab_size = len(vocab)
        id2word = vocab.id2word
    except NameError:
        print("[Demo Mode] 模拟生成 1000 个假概念进行测试...")
        vocab_size = 1000
        id2word = {i: f"concept_{i}" for i in range(1000)}
        dataset = torch.randint(0, 1000, (5000, 2))
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    manifold = TopologicalDomainSpace(num_domains=vocab_size, d_model=args.d_model, init_log_sigma=0.5).to(device)
    loss_fn = TopoMultiObjectiveLoss(delta_inc=0.05, delta_rep=2.0, alpha_tension=0.01, beta_rel=1.0).to(device)
    optimizer = torch.optim.AdamW(manifold.parameters(), lr=0.01)

    for epoch in range(1, args.epochs + 1):
        avg_loss = train_phase1_epoch(manifold, loader, optimizer, loss_fn, device, vocab_size)
        print(f"Epoch {epoch:02d}/{args.epochs} | Avg Physical Loss: {avg_loss:.4f}")

    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_path = os.path.join(args.output_dir, "topolm_phase1_emergence.pt")
    torch.save(manifold.state_dict(), ckpt_path)
    print(f"\n[+] 涌现流形已保存至: {ckpt_path}")
    with torch.no_grad(): radii = manifold.get_radius().cpu()
    export_and_plot_academic_figure(radii, id2word, args.results_dir)

if __name__ == "__main__":
    main()
