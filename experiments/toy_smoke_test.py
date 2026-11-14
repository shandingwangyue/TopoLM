"""Offline synthetic smoke experiment using the same model and trainer."""
import argparse
import sys
import os
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.topolm import TopoLM
from training.phase2_losses import TopoLMLoss
from training.trainer import TopoTrainer

def run_experiment(epochs=101):
    torch.manual_seed(42)
    model = TopoLM(vocab_size=100, num_domains=2, d_model=32, num_heads=2)
    trainer = TopoTrainer(model, TopoLMLoss(),
                          torch.optim.AdamW(model.parameters(), lr=0.01),
                          disjoint_pairs=[(0, 1)])
    streams = {d: torch.randint(d * 50, (d + 1) * 50, (4, 9)) for d in range(2)}
    inputs = {d: x[:, :-1] for d, x in streams.items()}
    targets = {d: x[:, 1:] for d, x in streams.items()}
    for epoch in range(epochs):
        metrics = trainer.train_step(inputs, targets)
        if epoch % 20 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:03d} | " + " | ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

if __name__ == "__main__":
    run_experiment()
