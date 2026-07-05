# Homeostatic-Transformer
"An experiment with homeostatic regulation in transformers."
# Homeostatic-Transformer  
*An experiment with homeostatic regulation in transformers.*

[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

---

## 📌 Overview

**Homeostatic-Transformer** is a research prototype that introduces **homeostatic regulation** into transformer layers.  
Each layer maintains a **temperature** (information load) and an **amnesia** (context forgetting), dynamically modulating hidden states.

✅ **Key results on TinyStories (5k samples):**
- **Higher diversity** (+1.6%)
- **More verbs** (+22% → richer storytelling)
- **Interpretable internal state** via "heartbeat" visualization

---

## 🧠 Why Homeostasis?

Biological brains use **homeostatic plasticity** to prevent over‑excitation and consolidate memory.  
Standard transformers treat every token identically. Homeostatic-Transformer adapts its processing intensity based on contextual stress, mimicking this biological mechanism.

---

## 🏗️ Architecture
Embedding → [HomeostaticLayer × N] → Output
├─ Attention + FFN
└─ HomeostaticModule
├─ Temperature (τ)
└─ Amnesia (α)

**HomeostaticModule** (per layer):
- **Information density** – cosine similarity between query (last token) and keys (mean context)
- **Temperature update** – learned gate + drift → clamped to [0.1, 5.0]
- **Amnesia gate** – activates when τ > critical threshold (default 2.0)
- **Output modulation** – `x * exp(-τ / 10)` (soft exponential decay)

---

## 📊 Experiment

| Model | Diversity ↑ | Repeat ↓ | Verbs ↑ |
|-------|-------------|----------|---------|
| Standard Transformer | 0.750 | 0.023 | 9 |
| **Homeostatic-Transformer** | **0.762** | 0.024 | **11** |

**Example generation:**
- **Standard:** *"Once upon a time, there was a little girl named Lily. ... her mom gave her a loud noise on the comet."*
- **Homeostatic:** *"Once upon a time, there was a little fish named Tom. Kitty loved to play with his friends. ... They played together in the seek."*

Homeostatic text shows more active characters and narrative structure.

---

## 💓 Heartbeat Visualization

The temperature of a layer can be plotted word‑by‑word, revealing the model's "pulse":
Word:     Once  upon  a  time  ,  a  little  girl  named  Lily  went  to  the  forest  .
Temp:     0.50  0.45 0.48 0.62 0.58 0.55 0.72  0.95  1.20  1.15  0.88 0.65 0.52  0.48

Peaks correspond to surprising or high‑information words.  
If the temperature exceeds the critical threshold, **amnesia accumulates**, protecting the model from overload.

*(See the notebook for the actual heartbeat plot.)*

---

## 🚀 Quick Start

Run everything in your browser – **no local GPU needed**.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

**Requirements:** `torch`, `datasets`, `tokenizers`, `matplotlib`

```python
from homeostatic_transformer import HomeostaticTransformer

model = HomeostaticTransformer(
    vocab_size=5000,
    embed_dim=128,
    num_layers=3,
    critical_temp=2.0
)

# Train & visualize – see the notebook
.
├── homeostatic_transformer.py   # Model implementation
├── train_and_evaluate.ipynb     # Full experiment notebook
├── heartbeat.png                # Example pulse plot
└── README.md
Novelty

· Learned homeostasis – temperature and amnesia are not hyperparameters but dynamic, trainable states.
· Exponential modulation – smooth signal decay without hard thresholds.
· Continuous state – temperature persists across tokens, forming a "load history".
· Interpretable – single scalar per layer shows model's internal stress.

---

📈 Future Work

· Scale to full TinyStories (2M+ stories)
· Integrate “sleep” phases into training (periodic resets)
· Apply to continual learning & long‑form dialog

---
@misc{homeostatictransformer2025,
  author = {Your Name},
  title = {Homeostatic-Transformer: An experiment with homeostatic regulation in transformers},
  year = {2025},
  note = {GitHub repository}
