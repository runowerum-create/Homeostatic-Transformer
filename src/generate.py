import torch
import torch.nn.functional as F
from collections import Counter
import numpy as np

def generate_text(model, tokenizer, prompt, max_new=200, temperature=0.8, is_homeo=False, device='cuda'):
    model.eval()
    input_ids = torch.tensor([tokenizer.encode(prompt).ids], device=device)
    generated = input_ids
    if is_homeo:
        temps = [torch.ones(1,1,device=device) for _ in range(model.num_layers)]
        amnesias = [torch.zeros(1,1,device=device) for _ in range(model.num_layers)]
    with torch.no_grad():
        for _ in range(max_new):
            if is_homeo:
                logits, temps, amnesias, _ = model(generated, temps, amnesias, return_hidden=True)
            else:
                logits, _ = model(generated, return_hidden=True)
            next_logits = logits[0, -1, :].clone()
            next_logits[0] = float('-inf')
            probs = F.softmax(next_logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, 1).unsqueeze(1)
            generated = torch.cat([generated, next_token], dim=1)
    return tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)

def ngram_diversity(text, n=3):
    tokens = text.split()
    if len(tokens) < n: return 0.0
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    return len(set(ngrams)) / len(ngrams) if ngrams else 0.0

def repeat_rate(text):
    tokens = text.split()
    if not tokens: return 0.0
    counts = Counter(tokens)
    repeated = sum(c for t, c in counts.items() if c > 1)
    return repeated / len(tokens)

VERBS = {"run","play","find","look","see","go","come","move","jump","walk","sit","stand","eat","drink","talk","say","tell","ask","think","know","want","need","like","love","hate","feel","become","get","make","take","explore","shout","cry","smile","laugh","agree","stay","leave","hold","point","answer"}

def count_verbs(text):
    clean = text.replace("Ġ", " ").lower().split()
    return sum(1 for w in clean if w in VERBS)
