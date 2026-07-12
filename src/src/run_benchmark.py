# =============================================================================
# BENCHMARK RUNNER (run_benchmark.py)
# Automated evaluation script comparing the Original and Causal models.
# Location: src/src/ next to the original experiment_runner.py
# 
# Note: The AI collaborator's guidelines and insights are integrated here.
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import re

# Import the original components and metric functions from the baseline
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

# Import our robust causal extension
from causal_extension import CausalPulseTransformer

print(f"\n[System] Benchmark infrastructure initialized. Device: {device}")
print(f"Tokenizer vocab size: {vocab_size}")

# -----------------------------------------------------------------------------
# 1. MODEL ASSEMBLY AND TRAINING
# -----------------------------------------------------------------------------

# Step A: Train the ORIGINAL model (the author's simplified baseline loop)
print("\n=== Step 1: Training Original PulseTransformer (Baseline) ===")
original_model = PulseTransformer(vocab_size=vocab_size, critical_temp=2.0).to(device)
original_model = train_model(original_model, "Original Pulse Transformer (GitHub Baseline)", epochs=5)

# Step B: Wrap a fresh baseline model into our Causal Extension Adapter
print("\n=== Step 2: Training CausalPulseTransformer (Our Improved Extension) ===")
fresh_base_model = PulseTransformer(vocab_size=vocab_size, critical_temp=2.0).to(device)
causal_model = CausalPulseTransformer(fresh_base_model).to(device)
causal_model = train_model(causal_model, "Causal Pulse Transformer (Our Extension)", epochs=5)

# -----------------------------------------------------------------------------
# 2. TEXT GENERATION AND METRICS EVALUATION
# -----------------------------------------------------------------------------
print("\n=== Step 3: Text Generation & Metrics Evaluation ===")
prompt = "Once upon a time"
start_ids = tokenizer.encode(prompt).ids

models_to_test = [
    ("Original Pulse (GitHub)", original_model),
    ("Causal Pulse (Extension)", causal_model)
]

benchmark_results = []

for name, model in models_to_test:
    print(f"\nGenerating text for {name}...")
    # The extension adapter shares an identical signature, ensuring seamless execution
    generated_text = generate_text(model, start_ids, max_len=40, is_pulse=True)
    
    # Calculate textual indicators (Diversity, Repeat Rate, Verbs)
    div, rep, vrb = calc_metrics(generated_text)
    benchmark_results.append((name, div, rep, vrb, generated_text))
    
    print(f" Sample output: {generated_text[:140]}...")

# -----------------------------------------------------------------------------
# 3. FINAL RESULTS BENCHMARK TABLE
# -----------------------------------------------------------------------------
print("\n" + "="*85)
print(f"{'Model Architecture':<30} | {'Diversity 📈':<12} | {'Repeat 📉':<10} | {'Verbs ⚙️':<8}")
print("-"*85)

for name, div, rep, vrb, _ in benchmark_results:
    print(f"{name:<30} | {div:.3f} | {rep:.3f} | {vrb:<8}")

print("="*85)

print("\n[Success] Anthropic-style evaluation complete! Final results compiled.")
