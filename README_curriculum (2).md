<div align="center">

# 📐 Token-Aware Curriculum Learning for LLM Pretraining

**A modular framework combining token-based LR scheduling, online curriculum filtering, RL-driven data selection, and multi-domain mixture for more efficient LLM pretraining**

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white) ![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white) ![HuggingFace](https://img.shields.io/badge/HuggingFace-transformers-FFD21E?logo=huggingface&logoColor=white) ![License](https://img.shields.io/badge/License-MIT-22c55e) ![Status](https://img.shields.io/badge/Status-active-brightgreen)

</div>

---


## 🔍 Overview

Standard LLM pretraining schedulers treat all steps equally — but not all tokens are equal.
This project explores a **token-aware training framework** that adapts four interacting axes:

1. **Learning rate** scales with *tokens processed*, not optimizer steps
2. **Curriculum** filters training samples dynamically by model-perceived difficulty
3. **Data selection** uses a lightweight RL policy to prioritize high-signal batches
4. **Domain mixture** samples heterogeneous datasets with learned or fixed weights

> **Why it matters:** Step-based schedulers under-utilize easy samples early and
> over-stress hard ones late. Token-aware scheduling naturally aligns compute with
> information density, improving sample efficiency without changing the model architecture.

---

## 🧩 Components

| Module | File | Description |
|---|---|---|
| Token LR Scheduler | `scheduler_token.py` | LR decay indexed by tokens seen, not steps |
| Online Curriculum | `online_curiculum.py` | Filters batches by per-sample loss threshold |
| RL Data Selector | `rl_selector.py` | Policy network that scores and selects batches |
| Domain Mixture | `mixture.py` | Weighted sampling across heterogeneous datasets |
| Trainer | `trainer.py` | Training loop integrating all components |

---

## 🏗️ Architecture

```
token-aware-curriculum-v2/
├── trainer.py              # Main training loop — integrates all components
├── scheduler_token.py      # Token-indexed learning rate scheduler
├── online_curiculum.py     # Online difficulty filtering (per-sample loss)
├── rl_selector.py          # RL policy for data selection
└── mixture.py              # Multi-domain dataset mixture with weights
```

**Training flow:**

```
Batch ──► DatasetMixture.sample()
              │
              ▼
         RLSelector.select()      ← scores batches, keeps high-signal ones
              │
              ▼
         OnlineCurriculum.filter_batch()   ← removes samples above difficulty threshold
              │
              ▼
         train_step()             ← forward + backward
              │
              ▼
         TokenScheduler.step(tokens)       ← LR update indexed by token count
```

---

## ⚙️ Installation

```bash
git clone https://github.com/Iamyulx/token-aware-curriculum-v2.git
cd token-aware-curriculum-v2
pip install torch transformers
```

---

## 🚀 Quickstart

```python
from transformers import AutoModelForCausalLM
from scheduler_token import TokenAwareScheduler
from online_curiculum import OnlineCurriculum
from rl_selector import RLSelector
from mixture import DatasetMixture
from trainer import train_step

# 1. Load model
model = AutoModelForCausalLM.from_pretrained("gpt2")
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

# 2. Token-aware scheduler
scheduler = TokenAwareScheduler(optimizer, total_tokens=1_000_000_000)

# 3. Curriculum + RL selector
curriculum = OnlineCurriculum(model)
selector   = RLSelector(input_dim=768)

# 4. Multi-domain mixture  (weights are normalized internally)
mixture = DatasetMixture(
    datasets=[code_dataset, web_dataset, books_dataset],
    weights=[0.4, 0.4, 0.2],
)

# 5. Training loop
for step in range(num_steps):
    batch  = mixture.sample()
    batch  = selector.select(batch)
    batch  = curriculum.filter_batch(batch, threshold=2.5)
    loss   = train_step(batch, model, optimizer, scheduler)
```

---

## 📊 Results

> ⚠️ **Placeholder values.** Run your experiments and update with real numbers.

| Ablation | Improvement |
|---|---|
| Token LR vs. step LR (perplexity ↓) | ~12% |
| Online curriculum vs. random filtering (perplexity ↓) | ~8% |
| RL selector vs. uniform sampling (throughput ↑) | ~15% |
| Multi-domain mixture vs. single-domain (generalization ↑) | ~6% |

---

## 🔬 Design Decisions

**Why token-based scheduling instead of step-based?**
Batch sizes vary across domains and dynamic filtering changes effective batch size per step.
Token-based scheduling keeps LR decay consistent regardless of how many samples were filtered.

**Why online curriculum instead of pre-sorted data?**
Pre-sorting requires a full dataset pass before training. Online filtering adapts to the
model's *current* knowledge state, which changes every step — making static sorting suboptimal
after the first few thousand steps.

**Why RL for data selection?**
A learned selector can discover non-obvious correlations between sample characteristics and
downstream loss improvement, beyond what a simple loss threshold can capture.

---

## ⚠️ Known Issues & Improvements

- `online_curiculum.py` has a typo in the filename (`curiculum` → `curriculum`) — rename in a future refactor
- `trainer.py` currently ignores `rl_selector` even when passed — wiring is pending
- No `requirements.txt` or `pyproject.toml` — add before publishing
- Flat file structure — consider `src/` layout for installability

---

## 📚 Related Work

- Bengio et al. (2009) — [Curriculum Learning](https://dl.acm.org/doi/10.1145/1553374.1553380)
- Graves et al. (2017) — [Automated Curriculum Learning via Absolute Error](https://arxiv.org/abs/1704.03732)
- Hoffmann et al. (2022) — [Training Compute-Optimal LLMs (Chinchilla)](https://arxiv.org/abs/2203.15556)
- Xie et al. (2023) — [DoReMi: Optimizing Data Mixtures](https://arxiv.org/abs/2305.10429)

---

## 📄 License

MIT © [Iamyulx](https://github.com/Iamyulx)
