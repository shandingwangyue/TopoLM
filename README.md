# TopoLM: Topological Manifold Language Modeling
**Zero-Hallucination Neuro-Symbolic AI via Spontaneous 9-Level Ontological Emergence**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Author**: Li Peng (Independent Researcher)  
**Contact**: tonyli288@gmail.com

---

## 🛑 The Problem: The Mathematical Inevitability of Hallucinations
Contemporary Large Language Models (LLMs) live in a **flat, zero-dimensional point-vector space**. Because standard self-attention evaluates token relationships via scaled dot-products, it can only measure **symmetric geometric similarity** ($Sim(A,B) = Sim(B,A)$). 

However, physical reality and deductive logic are inherently **asymmetric**: 
* A Husky $\subset$ Dog, but Dog $\not\subset$ Husky. 
* Fire $\perp$ Water (Absolute Mutual Exclusivity).

Forcing asymmetric logic into a symmetric, volumeless latent space inevitably leads to conceptual overlapping. This is why LLMs suffer from structural hallucinations and fail to maintain invariant physical laws—it is a mathematical destiny, not just a data deficit.

---

## 🚀 The Solution: TopoLM
**TopoLM** resolves this by restructuring the token space into a **Topological Manifold**. Instead of point vectors, every concept is parameterized as a multivariate Gaussian domain $\mathcal{N}(\mu, \Sigma)$ with an explicit **physical radius** and **spatial volume** using 2-Wasserstein geometry.

By permanently freezing the emergent spatial parameters as an immutable physical backbone ($\nabla_{\mu,\sigma} = 0$), TopoLM introduces an inverted spatial index to gather only topologically valid keys, reducing dense $\mathcal{O}(N^2)$ attention to **$\mathcal{O}(NK)$ Sparse-Indexed Attention**.

---

## 🛠️ Clean Modular Architecture

The repository has been heavily refactored for industrial-grade PyTorch deployment:

```text
TopoLM/
├── configs/
│   └── base.yaml                        # Hyperparameters & Topo-Domain Configurations
├── models/                              # [Core Architecture]
│   ├── geometry.py                      # 2-Wasserstein underlying mathematics
│   ├── domain_space.py                  # TopologicalDomainSpace (Manifolds, Mahalanobis distances)
│   ├── domain_router.py                 # Spatial Inverted Index Router for O(NK) Sparse Attention
│   ├── spatial_attention.py             # Dense/Sparse Topological Attention implementations
│   ├── constraint_layer.py              # Relative Voronoi output gating
│   └── topolm.py                        # Unified TopoLM Backbone
├── training/                            # [Multi-Objective Physics & Language Engines]
│   ├── phase1_loss.py                   # Pure Physical Emergence Loss (Inclusion, Repulsion, Tension)
│   ├── phase2_losses.py                 # Language + Volume Conservation NLL Loss
│   └── trainer.py                       # Unified Training Scheduler
└── experiments/                         # [Execution Pipelines & Ironclad Proofs]
    ├── run_phase1_emergence.py          # Spontaneous 9-Level Emergence (Sec 9.1)
    ├── run_phase2_brown.py              # Frozen-Backbone Language Fleshing (Sec 9.2)
    ├── demo_syllogism_and_phase2.py     # Zero-Shot Syllogism & Hallucination Block (Sec 8.1)
    ├── evaluate_e3_audit.py             # E3 Integrity Audit (Table 3)
    ├── benchmark_complexity.py          # O(NK) Complexity Benchmark (Table 1)
    ├── train_sandbox_domains.py         # Multi-Domain Topological Isolation
    └── toy_smoke_test.py                # Synthetic Gradient Check & Smoke Test
```

---

## 🔬 The Ironclad Proofs (Paper to Code Mapping)

This repository is designed to be a 1-to-1 executable replication of the claims made in our research paper. Every core theorem and table can be instantly verified via the **7 execution scripts** in the `experiments/` directory.

### Part 1: The 5 Core Demonstrations

| Paper Section | Claim / Demonstration | Executable Script |
| :--- | :--- | :--- |
| **Sec 9.1** | **Spontaneous 9-Level Emergence**: 60k unlabelled concepts autonomously form a power-law pyramid via physical surface tension and repulsion. | `python experiments/run_phase1_emergence.py` |
| **Sec 9.2** | **Frozen-Backbone Language Fleshing**: Causal language modeling on the Brown corpus with strictly zero topological drift ($\nabla_{\mu,\sigma} = 0$). | `python experiments/run_phase2_brown.py` |
| **Sec 8.1** | **Zero-Shot Syllogism & Hallucination Block**: Deduce `Husky ⊂ Requires Energy` without direct training, and physically block paradoxical tokens. | `python experiments/demo_syllogism_and_phase2.py` |
| **Table 3** | **E3 Integrity Audit**: Quantitative Mahalanobis distance and absolute containment margins between disjoint/subset concepts. | `python experiments/evaluate_e3_audit.py` |
| **Table 1** | **$\mathcal{O}(NK)$ Complexity Benchmark**: Real-time GPU VRAM verification proving the linear scaling of Spatial-Indexed Attention up to 128k context. | `python experiments/benchmark_complexity.py` |

### Part 2: Sandbox & Development Tooling

| Purpose | Description | Executable Script |
| :--- | :--- | :--- |
| **Sandbox Pipeline** | **Multi-Domain Topological Isolation**: Train the causal model on constrained domains (Physics Laws vs. Fantasy Novels) and evaluate the Voronoi output gating. | `python experiments/train_sandbox_domains.py` |
| **Dev Smoke Test** | **Synthetic Gradient Check**: Verify the full forward/backward pass and multi-objective loss computation on a synthetic 100-token vocabulary. | `python experiments/toy_smoke_test.py` |

---

## ⚡ Quick Start

**1. Install Dependencies**
```bash
python -m pip install -r requirements.txt
```

**2. Verify VRAM Scaling (O(N²) vs O(NK))**
Run the benchmark to see the massive memory reduction achieved by TopoLM's Spatial-Indexed Attention:
```bash
python experiments/benchmark_complexity.py
```

**3. Run the Zero-Shot Syllogism Deduction**
Witness how the 2-Wasserstein geometric manifold autonomously deduces logical chains and intercepts paradoxical hallucinations via absolute containment margins:
```bash
python experiments/demo_syllogism_and_phase2.py
```

---

## 📖 Citation & Academic Release
The full preprint detailing the underlying theoretical proofs, 60k WordNet emergence scale-up, and 128k long-context complexity analysis is currently undergoing independent peer review / arXiv preprint preparation. 

If this repository inspires your work in Neuro-Symbolic AI, AGI World Models, or Hallucination Mitigation, please cite this GitHub repository.
