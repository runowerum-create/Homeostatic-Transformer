# =============================================================================
# BENCHMARK RUNNER (run_benchmark.py)
# Скрипт для автоматического сравнения оригинальной и каузальной моделей.
# Поместить в папку: src/src/ рядом с оригинальным experiment_runner.py
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import re

# Импортируем оригинальные компоненты и функции расчета метрик
from experiment_runner import (
    device, 
    tokenizer, 
    vocab_size, 
    X, 
    Y, 
    train_model, 
    generate_text, 
    calc_metrics,
    PulseTransformer
)

# Импортируем наше честное каузальное расширение
from causal_extension import CausalPulseTransformer

print(f"\n[Запуск] Инфраструктура готова. Устройство: {device}")
print(f"Размер словаря оригинального токенизатора: {vocab_size}")

# -----------------------------------------------------------------------------
# 1. СБОРКА И ОБУЧЕНИЕ МОДЕЛЕЙ
# -----------------------------------------------------------------------------

# Шаг А: Создаем и обучаем ОРИГИНАЛЬНУЮ модель (упрощенный causal-loop автора)
print("\n=== Шаг 1: Обучение оригинальной PulseTransformer ===")
original_model = PulseTransformer(vocab_size=vocab_size, critical_temp=2.0).to(device)
original_model = train_model(original_model, "Original Pulse Transformer (GitHub Baseline)", epochs=5)

# Шаг Б: Создаем новую чистую оригинальную модель и оборачиваем в наш Causal-Адаптер
print("\n=== Шаг 2: Обучение модифицированной CausalPulseTransformer ===")
fresh_base_model = PulseTransformer(vocab_size=vocab_size, critical_temp=2.0).to(device)
causal_model = CausalPulseTransformer(fresh_base_model).to(device)
causal_model = train_model(causal_model, "Causal Pulse Transformer (Our Extension)", epochs=5)

# -----------------------------------------------------------------------------
# 2. ГЕНЕРАЦИЯ ТЕКСТА И СБОР МЕТРИК
# -----------------------------------------------------------------------------
print("\n=== Шаг 3: Тестирование генерации и сбор метрик ===")
prompt = "Once upon a time"
start_ids = tokenizer.encode(prompt).ids

models_to_test = [
    ("Original Pulse (GitHub)", original_model),
    ("Causal Pulse (Extension)", causal_model)
]

benchmark_results = []

for name, model in models_to_test:
    print(f"\nГенерация текста для модели {name}...")
    # generate_text внутри вызывает model(generated, temps, amnesias) 
    # Интерфейс нашего адаптера полностью идентичен, поэтому вызов отработает бесшовно
    generated_text = generate_text(model, start_ids, max_len=40, is_pulse=True)
    
    # Считаем метрики текста (Diversity, Repeat, Verbs)
    div, rep, vrb = calc_metrics(generated_text)
    benchmark_results.append((name, div, rep, vrb, generated_text))
    
    print(f" Результат: {generated_text[:140]}...")

# -----------------------------------------------------------------------------
# 3. СВЕДЕНИЕ РЕЗУЛЬТАТОВ В ИТОГОВУЮ ТАБЛИЦУ
# -----------------------------------------------------------------------------
print("\n" + "="*85)
print(f"{'Архитектура Модели':<30} | {'Diversity 📈':<12} | {'Repeat 📉':<10} | {'Verbs ⚙️':<8}")
print("-"*85)

for name, div, rep, vrb, _ in benchmark_results:
    print(f"{name:<30} | {div:.3f} | {rep:.3f} | {vrb:<8}")

print("="*85)

print("\n[Успешно] Эксперимент Anthropic-стиля завершен! Финальные логи сохранены.")
