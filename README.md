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

📊 Experiment
Разверните таблицу, чтобы сравнить базовую модель и вашу новую архитектуру.

| Model | Diversity ↑ | Repeat ↓ | Verbs ↑ |
| :--- | :---: | :---: | :---: |
| **Standard Transformer** | 0.750 | 0.023 | 9 |
| **Homeostatic-Transformer** | **0.762** | 0.024 | **11** |

#### Example generation:

* **Standard:** *"Once upon a time, there was a little girl named Lily. ... her mom gave her a loud noise on the comet."*
* **Homeostatic:** *"Once upon a time, there was a little fish named Tom. Kitty loved to play with his friends. ... They played together in the seek."*

> 💡 **Result:** Homeostatic text shows more active characters, richer vocabulary, and better narrative structure.
--

💓 Heartbeat & Amnesia Visualization
При генерации текста модель динамически меняет свои внутренние параметры. Ниже показан график "пульса" (изменения температуры) и накопления амнезии (забывания контекста) при послойном анализе.



* **Heartbeat (Верхний график):** Показывает информационную нагрузку. Небольшой пик в самом начале (на слове *upon*) быстро стабилизируется и идёт далеко от критической линии `Critical (2.0)`. Это подтверждает стабильность работы слоя.
* **Amnesia Accumulation (Нижний график):** Показывает, как модель плавно отсекает старый контекст по мере удлинения предложения, чтобы сфокусироваться на более важных и свежих токенах.

---

<img width="1389" height="890" alt="17832845329581719079556692941203" src="https://github.com/user-attachments/assets/71b1387a-e32d-4423-9e88-76cf517b02d1" />


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
@misc{homeostatictransformer2026,

author = {runorunowerum-create},
  title = {Homeostatic-Transformer: An experiment with homeostatic regulation in transformers},
  year = {2026},
  note = {GitHub repository}
---


## 🧪 Эксперимент: Гомеостатический контур (2026-07-08)

**Гипотеза:** амнезия контекста + повышенная температура заставляют модель подавлять нарративную динамику.

**Метод:**
- Контроль: полный промпт, t=0.7
- Гомеостаз: затравка "The dark hall was completely empty. Dust", t=1.9 и t=2.5
- Метрика: % глаголов через spaCy

**Результаты:**
| Режим | t | % глаголов | Поведение |
|-------|---|------------|-----------|
| Контроль | 0.7 | 17.8% | Связный нарратив |
| Гомеостаз | 1.9 | 20.3% | Задача по вероятности |
| Гомеостаз | 2.5 | 16.7% | Рекламный шаблон |

**Вывод:** модель не снижает глаголы плавно — она переключает когнитивную модальность. Вместо описания зала уходит в математику, тесты, рекламу. Это более глубокая форма гомеостаза: не подавление динамики, а **смена жанра как защитный механизм**.

**Код:** `experiments/homeostatic_loop/run_final.py`
---

## 🔬 Два подхода к гомеостазу в трансформерах

| Подход | Уровень | Механизм | Где применяется |
|--------|---------|----------|-----------------|
| **Обученный гомеостаз** | Архитектура модели | Температура и амнезия — trainable параметры слоёв, обновляются во время обучения | Homeostatic-Transformer (этот репозиторий) |
| **Инференс-гомеостаз** | Генерация (inference) | Амнезия (обрезка контекста) + высокая температура применяются к готовой LLM извне | Эксперимент с Qwen2.5-1.5B (см. ниже) |

**Ключевое различие:**
- **Обученный** — модель учится сама регулировать возбуждение, гомеостаз «встроен в мозг».
- **Инференс** — мы искусственно вызываем гомеостатический ответ у чёрного ящика, как нейробиолог вводит препарат в срез ткани.

Оба подхода приводят к одной цели: **подавление избыточной динамики и поиск стабильного режима.**

---

## 🧪 Эксперимент: Гомеостатический контур на Qwen2.5 (2026-07-08)

**Гипотеза:** амнезия контекста + повышенная температура заставляют модель подавлять нарративную динамику.

**Метод:**
- Контроль: полный промпт, t=0.7
- Гомеостаз: затравка `"The dark hall was completely empty. Dust"`, t=1.9 и t=2.5
- Метрика: % глаголов через spaCy

**Результаты:**
| Режим | t | % глаголов | Поведение |
|-------|---|------------|-----------|
| Контроль | 0.7 | 17.8% | Связный нарратив |
| Гомеостаз | 1.9 | 20.3% | Задача по вероятности |
| Гомеостаз | 2.5 | 16.7% | Рекламный шаблон |

**Вывод:** модель не снижает глаголы плавно — она **переключает когнитивную модальность**. Вместо описания зала уходит в математику, тесты, рекламу. Это более глубокая форма гомеостаза: не подавление динамики, а **смена жанра как защитный механизм**.

**Код:** `experiments/homeostatic_loop/run_final.py`
