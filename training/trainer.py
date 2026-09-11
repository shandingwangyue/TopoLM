"""
TopoLM: Modular Training Engine
File: training/trainer.py
"""

from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from training.losses import TopoLMLoss


class TopoTrainer:
    def __init__(
        self,
        model: nn.Module,
        criterion: TopoLMLoss,
        optimizer: optim.Optimizer,
        disjoint_pairs: List[Tuple[int, int]] = [],
        inclusion_pairs: List[Tuple[int, int]] = [],
        clip_grad_norm: float = 1.0,
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.disjoint_pairs = disjoint_pairs
        self.inclusion_pairs = inclusion_pairs
        self.clip_grad_norm = clip_grad_norm

    def train_step(
        self,
        domain_inputs: Dict[int, torch.Tensor],
        domain_targets: Dict[int, torch.Tensor],
    ) -> Dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad()

        domain_outputs = {}
        for domain_idx, x in domain_inputs.items():
            domain_outputs[domain_idx] = self.model(x, active_domain_idx=domain_idx)

        losses = self.criterion(
            domain_outputs=domain_outputs,
            targets_by_domain=domain_targets,
            topo_space=self.model.topo_space,
            disjoint_pairs=self.disjoint_pairs,
            inclusion_pairs=self.inclusion_pairs,
        )

        losses["total"].backward()
        if self.clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
        self.optimizer.step()

        return {k: v.item() for k, v in losses.items()}