# TopoLM: Topological Domain Language Model

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

**Author:** Li Peng 
**Paper:** *Toward Artificial General Intelligence: A Cognitive Architecture Based on 9-Level Token Stratification and Topological Domain Embeddings*

TopoLM is a neuro-symbolic cognitive architecture designed to overcome the structural limitations of standard flat-sequence Large Language Models (LLMs). By replacing one-dimensional vector points with **Multivariate Gaussian Topological Domains**, TopoLM introduces explicit spatial logic (inclusion, mutual exclusivity, and boundaries) directly into the attention mechanism.

## 🚀 Key Innovations

*   **9-Level Token Stratification:** Vocabulary is segmented into Macro Domains (Axioms/Physics), Meso Domains (Fields/Context), and Micro Entities (Atomic Vectors).
*   **Topological Spatial Attention:** Bypasses the $O(N^2)$ bottleneck. The model selectively masks out tokens that fall outside the probability density function (PDF) of the active Gaussian domain.
*   **Immutable World Model Priors:** Top-level physical and logical rules are frozen post-training. The model rejects hallucinations not through RLHF safety filters, but through sheer "computational friction" at the geometric level.

## ⚙️ How It Works (Case Studies)

### 1. Epistemic Conviction (Axiomatic Refusal)
If prompted to generate a "perpetual motion engine" operating at 100% efficiency, the system activates the `DOMAIN_PHYSICS` constraint. Because entropy reduction is geometrically disjoint from this domain manifold, the model experiences extreme gradient penalties and masks the attention to $-\infty$, executing a mathematically verified refusal.

### 2. Cross-Domain Decoupling
When generating literary content, such as a scene from the Xianxia novel 《十万里风雪客》 where the protagonist manipulates inner Qi to step on the void, the context router activates the `DOMAIN_XIANXIA` (Meso-Domain). The rigid physical covariance matrices are bypassed, allowing stylistic generation without polluting the foundational real-world physics weights.

## 📦 Installation & Usage

```bash
git clone [https://github.com/shandingwangyue/TopoLM.git](https://github.com/shandingwangyue/TopoLM.git)
cd TopoLM
pip install -r requirements.txt
