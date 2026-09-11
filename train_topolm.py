"""
TopoLM: Master Execution Pipeline
File: train_topolm.py
"""

import os
import yaml
import torch
import torch.optim as optim
from transformers import BertTokenizer

from models.topolm import TopoLM
from training.losses import TopoLMLoss
from training.trainer import TopoTrainer


def load_config(config_path="configs/base.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def create_dataset(tokenizer, text, seq_len):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    inputs, targets = [], []
    for i in range(len(tokens) - seq_len):
        inputs.append(tokens[i : i + seq_len])
        targets.append(tokens[i + 1 : i + seq_len + 1])
    return torch.tensor(inputs, dtype=torch.long), torch.tensor(targets, dtype=torch.long)


@torch.no_grad()
def generate_text(model, tokenizer, prompt, domain_idx, max_tokens=16, device="cuda"):
    model.eval()
    input_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(device)

    for _ in range(max_tokens):
        context = input_ids[:, -model.max_seq_len:]
        # 启用输出端 Voronoi 相对拓扑门控约束
        outputs = model(context, active_domain_idx=domain_idx, apply_constraint=True)
        next_token = outputs["logits"][:, -1, :].argmax(dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_token], dim=1)

    return tokenizer.decode(input_ids[0].tolist())


def main():
    cfg = load_config()
    device = torch.device(cfg["training"]["device"] if torch.cuda.is_available() else "cpu")
    print(f"Executing TopoLM Pipeline on: {device}")

    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    vocab_size = len(tokenizer)

    # 1. 实例化拓扑模型
    model = TopoLM(
        vocab_size=vocab_size,
        num_domains=cfg["model"]["num_domains"],
        d_model=cfg["model"]["d_model"],
        num_heads=cfg["model"]["num_heads"],
        max_seq_len=cfg["model"]["max_seq_len"],
        init_log_sigma=cfg["model"]["init_log_sigma"],
        sparse_mode=False,
    ).to(device)

    # 2. 实例化联合多目标损失函数
    criterion = TopoLMLoss(
        alpha_assign=cfg["loss"]["alpha_assign"],
        alpha_nll=cfg["loss"]["alpha_nll"],
        beta_rel=cfg["loss"]["beta_rel"],
        beta_inc=cfg["loss"]["beta_inc"],
        margin=cfg["loss"]["margin"],
    )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    trainer = TopoTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        disjoint_pairs=[(cfg["domains"]["PHYSICS"], cfg["domains"]["FANTASY"])],
    )

    # 3. 准备微型双语料基准
    text_phys = "孤立系统的熵在不伴随外界作功的情况下永远不会减少，热力学定律是物理学的重要基本规律。"
    text_fant = "十万里风雪客运转真气，踏破虚空，不借助飞剑便直上云霄，这是独属于他的仙侠大道。"

    seq_len = cfg["training"]["seq_len"]
    x_p, y_p = create_dataset(tokenizer, text_phys, seq_len)
    x_f, y_f = create_dataset(tokenizer, text_fant, seq_len)

    domain_inputs = {cfg["domains"]["PHYSICS"]: x_p.to(device), cfg["domains"]["FANTASY"]: x_f.to(device)}
    domain_targets = {cfg["domains"]["PHYSICS"]: y_p.to(device), cfg["domains"]["FANTASY"]: y_f.to(device)}

    print(f"Beginning Training: {cfg['training']['epochs']} Epochs | Vocab: {vocab_size} | SeqLen: {seq_len}\n")

    for epoch in range(cfg["training"]["epochs"]):
        metrics = trainer.train_step(domain_inputs, domain_targets)

        if epoch % 20 == 0 or epoch == cfg["training"]["epochs"] - 1:
            dist = model.topo_space.domain_centroid_distance(
                cfg["domains"]["PHYSICS"], cfg["domains"]["FANTASY"]
            ).item()
            sigmas = model.topo_space.get_sigma().mean(dim=1).cpu().tolist()

            print(
                f"Epoch {epoch:03d} | Total: {metrics['total']:.3f} | LM: {metrics['lm']:.3f} | "
                f"Assign: {metrics['assign']:.3f} | NLL: {metrics['nll']:.1f} | Rel: {metrics['relation']:.2f}"
            )
            print(f"           | Centroid Dist: {dist:.2f} | Sigma: [{sigmas[0]:.2f}, {sigmas[1]:.2f}]")
    # 保存训练好的拓扑参数与嵌入层
    torch.save(model.state_dict(), "topolm_weights.pt")
    print("Saved trained weights to topolm_weights.pt")
    # 4. 执行真实自回归生成验证
    print("\n" + "=" * 60)
    print("           Domain-Conditioned Rollout Verification")
    print("=" * 60)
    gen_fant = generate_text(model, tokenizer, "十万里风雪客", cfg["domains"]["FANTASY"], 14, device)
    gen_phys = generate_text(model, tokenizer, "孤立系统的熵", cfg["domains"]["PHYSICS"], 14, device)

    print(f"[Fantasy Domain] Prompt: '十万里风雪客' ->\n  Output: {gen_fant}\n")
    print(f"[Physics Domain] Prompt: '孤立系统的熵' ->\n  Output: {gen_phys}")
    print("=" * 60)


if __name__ == "__main__":
    main()