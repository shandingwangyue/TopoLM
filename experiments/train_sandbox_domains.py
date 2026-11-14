"""
TopoLM: Sandbox Multi-Domain Training & Voronoi Rollout Verification
File: experiments/train_sandbox_domains.py
"""

import os
import sys
import torch
import torch.optim as optim
from transformers import BertTokenizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.topolm import TopoLM
from training.phase2_losses import TopoLMLoss
from training.trainer import TopoTrainer


def load_config():
    return {
        "model": {"num_domains": 2, "d_model": 64, "num_heads": 4, "max_seq_len": 128, "init_log_sigma": 0.5, "sparse_mode": False},
        "loss": {"alpha_assign": 1.0, "alpha_nll": 0.05, "beta_rel": 0.5, "beta_inc": 0.5, "margin": 8.0},
        "training": {"device": "cpu", "lr": 3e-3, "weight_decay": 1e-2, "seq_len": 24, "epochs": 25, "clip_grad_norm": 1.0},
        "domains": {"PHYSICS": 0, "FANTASY": 1}
    }


def create_dataset(tokenizer, text, seq_len):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    inputs, targets = [], []
    for i in range(len(tokens) - seq_len):
        inputs.append(tokens[i : i + seq_len])
        targets.append(tokens[i + 1 : i + seq_len + 1])
    return torch.tensor(inputs, dtype=torch.long), torch.tensor(targets, dtype=torch.long)


@torch.no_grad()
def generate_text(model, tokenizer, prompt, domain_idx, max_tokens=16, device="cpu"):
    model.eval()
    input_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(device)
    for _ in range(max_tokens):
        context = input_ids[:, -model.max_seq_len:]
        outputs = model(context, active_domain_idx=domain_idx, apply_constraint=True)
        next_token = outputs["logits"][:, -1, :].argmax(dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_token], dim=1)
    return tokenizer.decode(input_ids[0].tolist())


def main():
    cfg = load_config()
    device = torch.device(cfg["training"]["device"] if torch.cuda.is_available() else "cpu")
    
    class DummyTokenizer:
        def __init__(self): self.vocab_size = 200
        def encode(self, t, add_special_tokens=False, return_tensors=None):
            res = [abs(hash(c)) % 200 for c in t]
            return torch.tensor([res]) if return_tensors == "pt" else res
        def decode(self, t): return "".join(chr(i + 100) for i in t)
        def __len__(self): return 200
    
    tokenizer = DummyTokenizer()
    vocab_size = len(tokenizer)

    model = TopoLM(
        vocab_size=vocab_size,
        num_domains=cfg["model"]["num_domains"],
        d_model=cfg["model"]["d_model"],
        num_heads=cfg["model"]["num_heads"],
        max_seq_len=cfg["model"]["max_seq_len"],
        init_log_sigma=cfg["model"]["init_log_sigma"],
        sparse_mode=cfg["model"].get("sparse_mode", False),
    ).to(device)

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
        clip_grad_norm=cfg["training"].get("clip_grad_norm", 1.0),
    )

    text_phys = "孤立系统的熵在不伴随外界作功的情况下永远不会减少" * 3
    text_fant = "十万里风雪客运转真气踏破虚空" * 3

    seq_len = cfg["training"]["seq_len"]
    x_p, y_p = create_dataset(tokenizer, text_phys, seq_len)
    x_f, y_f = create_dataset(tokenizer, text_fant, seq_len)

    domain_inputs = {cfg["domains"]["PHYSICS"]: x_p.to(device), cfg["domains"]["FANTASY"]: x_f.to(device)}
    domain_targets = {cfg["domains"]["PHYSICS"]: y_p.to(device), cfg["domains"]["FANTASY"]: y_f.to(device)}

    for epoch in range(cfg["training"]["epochs"]):
        metrics = trainer.train_step(domain_inputs, domain_targets)
        if epoch % 20 == 0 or epoch == cfg["training"]["epochs"] - 1:
            dist = model.topo_space.domain_centroid_distance(0, 1).item()
            sigmas = model.topo_space.get_sigma().mean(dim=1).cpu().tolist()
            print(
                f"Epoch {epoch:03d} | Total: {metrics['total']:.3f} | LM: {metrics['lm']:.3f} | "
                f"Assign: {metrics['assign']:.3f} | NLL: {metrics['nll']:.1f} | Rel: {metrics['relation']:.2f}"
            )

    gen_fant = generate_text(model, tokenizer, "十万里风雪客", cfg["domains"]["FANTASY"], 14, device)
    gen_phys = generate_text(model, tokenizer, "孤立系统的熵", cfg["domains"]["PHYSICS"], 14, device)
    print(f"[Fantasy Domain] -> {gen_fant}")
    print(f"[Physics Domain] -> {gen_phys}")

if __name__ == "__main__":
    main()
