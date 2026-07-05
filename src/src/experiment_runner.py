# =============================================================================
# PULSE TRANSFORMER — ФИНАЛЬНЫЙ ЗАПУСК (ВСЁ В ОДНОМ)
# Просто нажмите "Запустить" и ждите результат.
# =============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
import matplotlib.pyplot as plt
import math
import re
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Устройство: {device}")

# -----------------------------------------------------------------------------
# ВСЕ КОМПОНЕНТЫ МОДЕЛИ (RoPE, PulseModule, PulseTransformer и т.д.)
# -----------------------------------------------------------------------------
def precompute_rope_freqs(dim, max_seq_len=512, base=10000.0):
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq_len).float()
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)

class RotaryMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1, max_seq_len=512):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        cos, sin = precompute_rope_freqs(self.head_dim, max_seq_len)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    def forward(self, x, causal_mask=True):
        B, S, _ = x.shape
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        cos_cur = self.cos[:S].unsqueeze(0).unsqueeze(0)
        sin_cur = self.sin[:S].unsqueeze(0).unsqueeze(0)
        q, k = apply_rotary_pos_emb(q, k, cos_cur, sin_cur)
        scale = math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) / scale
        if causal_mask:
            mask = torch.triu(torch.ones(S, S, device=x.device) * float('-inf'), diagonal=1)
            attn = attn + mask
        attn = F.softmax(attn, dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, S, self.embed_dim)
        return self.out_proj(out)

class StandardTransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_heads=4, ff_dim=256, dropout=0.1, max_seq_len=512):
        super().__init__()
        self.attention = RotaryMultiheadAttention(embed_dim, num_heads, dropout, max_seq_len)
        self.ff = nn.Sequential(nn.Linear(embed_dim, ff_dim), nn.GELU(), nn.Linear(ff_dim, embed_dim))
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out = self.attention(x, causal_mask=True)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x

class PulseModule(nn.Module):
    def __init__(self, embed_dim, critical_temp=2.0, temp_scale=0.3):
        super().__init__()
        self.critical_temp = critical_temp
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.temp_gate = nn.Linear(embed_dim, 1)
        self.temp_scale = nn.Parameter(torch.tensor(temp_scale))
        nn.init.xavier_uniform_(self.temp_gate.weight, gain=1.5)

    def forward(self, x, prev_temp, prev_amnesia):
        q = self.query_proj(x[:, -1, :])
        k = self.key_proj(x).mean(dim=1)
        info_density = torch.sigmoid((q * k).sum(dim=-1, keepdim=True))
        delta_temp = torch.tanh(self.temp_gate(x[:, -1, :])) * 0.5
        new_temp = prev_temp + delta_temp * (info_density + 0.1)
        new_temp = torch.clamp(new_temp, min=0.1, max=5.0)
        amnesia_gate = torch.sigmoid(new_temp - self.critical_temp)
        new_amnesia = prev_amnesia * (1 - amnesia_gate) + amnesia_gate
        modulation = torch.exp(-new_temp.unsqueeze(1) / 10.0)
        return x * modulation, new_temp, new_amnesia

class PulseTransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_heads=4, ff_dim=256, critical_temp=2.0, temp_scale=0.3, dropout=0.1, max_seq_len=512):
        super().__init__()
        self.attention = RotaryMultiheadAttention(embed_dim, num_heads, dropout, max_seq_len)
        self.ff = nn.Sequential(nn.Linear(embed_dim, ff_dim), nn.GELU(), nn.Linear(ff_dim, embed_dim))
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.pulse = PulseModule(embed_dim, critical_temp, temp_scale)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, prev_temp, prev_amnesia):
        attn_out = self.attention(x, causal_mask=True)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return self.pulse(x, prev_temp, prev_amnesia)

class StandardTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_layers=3, num_heads=4, ff_dim=256, max_seq_len=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.layers = nn.ModuleList([
            StandardTransformerLayer(embed_dim, num_heads, ff_dim, max_seq_len=max_seq_len)
            for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        out = self.embedding(x)
        for layer in self.layers:
            out = layer(out)
        return self.fc_out(out)

class PulseTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_layers=3, num_heads=4, ff_dim=256,
                 critical_temp=2.0, temp_scale=0.3, max_seq_len=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.layers = nn.ModuleList([
            PulseTransformerLayer(embed_dim, num_heads, ff_dim, critical_temp, temp_scale, max_seq_len=max_seq_len)
            for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(embed_dim, vocab_size)
        self.num_layers = num_layers

    def forward(self, x, temps=None, amnesias=None):
        B = x.size(0)
        if temps is None:
            temps = [torch.ones(B, 1, device=x.device) for _ in range(self.num_layers)]
        if amnesias is None:
            amnesias = [torch.zeros(B, 1, device=x.device) for _ in range(self.num_layers)]
        out = self.embedding(x)
        new_temps, new_amnesias = [], []
        for i, layer in enumerate(self.layers):
            out, t, a = layer(out, temps[i], amnesias[i])
            new_temps.append(t)
            new_amnesias.append(a)
        return self.fc_out(out), new_temps, new_amnesias

# -----------------------------------------------------------------------------
# ДАННЫЕ И ТОКЕНИЗАТОР
# -----------------------------------------------------------------------------
print("Подготовка данных...")
dataset = load_dataset("roneneldan/TinyStories", split="train[:5000]")
tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
trainer = trainers.BpeTrainer(vocab_size=5000, special_tokens=["<pad>", "<unk>", "<s>", "</s>"])
tokenizer.train_from_iterator(dataset["text"], trainer)
vocab_size = tokenizer.get_vocab_size()

MAX_LEN = 64
tokenized = [tokenizer.encode(t).ids for t in dataset["text"]]
tokenized = [t[:MAX_LEN] for t in tokenized if len(t) > 5]
padded = [t + [0]*(MAX_LEN - len(t)) for t in tokenized]
X = torch.tensor([t[:-1] for t in padded], dtype=torch.long).to(device)
Y = torch.tensor([t[1:] for t in padded], dtype=torch.long).to(device)

# -----------------------------------------------------------------------------
# ОБУЧЕНИЕ
# -----------------------------------------------------------------------------
def train_model(model, name, epochs=5, lr=0.003):
    print(f"\nОбучение {name}...")
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        perm = torch.randperm(X.size(0))
        for i in range(0, X.size(0), 128):
            idx = perm[i:i+128]
            bx, by = X[idx], Y[idx]
            optimizer.zero_grad()
            if isinstance(model, PulseTransformer):
                logits, _, _ = model(bx)
            else:
                logits = model(bx)
            loss = criterion(logits.view(-1, vocab_size), by.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"  Эпоха {epoch+1}: loss={total_loss/(X.size(0)//128+1):.3f}")
    return model

baseline = StandardTransformer(vocab_size=vocab_size).to(device)
baseline = train_model(baseline, "Standard Transformer", epochs=5)

pulse = PulseTransformer(vocab_size=vocab_size, critical_temp=2.0).to(device)
pulse = train_model(pulse, "Pulse Transformer", epochs=5)

# -----------------------------------------------------------------------------
# ГЕНЕРАЦИЯ И МЕТРИКИ
# -----------------------------------------------------------------------------
def generate_text(model, prompt_ids, max_len=40, is_pulse=False):
    model.eval()
    generated = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    if is_pulse:
        temps = [torch.ones(1, 1, device=device) for _ in range(model.num_layers)]
        amnesias = [torch.zeros(1, 1, device=device) for _ in range(model.num_layers)]
    with torch.no_grad():
        for _ in range(max_len):
            if is_pulse:
                logits, temps, amnesias = model(generated, temps, amnesias)
            else:
                logits = model(generated)
            next_logits = logits[0, -1, :].clone()
            unk_id = tokenizer.token_to_id("<unk>")
            pad_id = tokenizer.token_to_id("<pad>")
            next_logits[unk_id] = float('-inf')
            next_logits[pad_id] = float('-inf')
            probs = F.softmax(next_logits / 0.9, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).unsqueeze(0)
            generated = torch.cat([generated, next_token], dim=1)
            if generated.size(1) > MAX_LEN:
                generated = generated[:, -MAX_LEN:]
    raw = tokenizer.decode(generated[0].tolist())
    clean = raw.replace('Ġ', ' ').replace('Ċ', '\n')
    clean = re.sub(r'[^\x00-\x7F]+', ' ', clean)
    return ' '.join(clean.split())

def calc_metrics(text):
    words = text.lower().split()
    if not words: return 0,0,0
    div = len(set(words))/len(words)
    bigrams = [tuple(words[i:i+2]) for i in range(len(words)-1)]
    rep = 1.0 - len(set(bigrams))/max(1,len(bigrams))
    vrb = sum(1 for w in words if w.endswith(('ed','ing','s','es','er')) or w in ('is','are','was','were','be','have','has','had','do','does','did','go','goes','went','say','said','tell','told'))
    return div, rep, vrb

print("\nГенерация текстов...")
prompt = "Once upon a time"
start_ids = tokenizer.encode(prompt).ids
results = []
for name, model, is_pulse in [("Standard Transformer", baseline, False), ("Pulse Transformer", pulse, True)]:
    text = generate_text(model, start_ids, 40, is_pulse)
    div, rep, vrb = calc_metrics(text)
    results.append((name, div, rep, vrb, text))
    print(f"\n{name}: Diversity={div:.3f}, Repeat={rep:.3f}, Verbs={vrb}")
    print(f"  {text[:200]}...")

# -----------------------------------------------------------------------------
# ТАБЛИЦА
# -----------------------------------------------------------------------------
print("\n" + "="*70)
print(f"{'Model':<25} | {'Diversity':<10} | {'Repeat':<8} | {'Verbs':<6}")
print("-"*70)
for name, div, rep, vrb, _ in results:
    print(f"{name:<25} | {div:.3f}      | {rep:.3f}    | {vrb}")
print("="*70)

# -----------------------------------------------------------------------------
# ПУЛЬС (HEARTBEAT)
# -----------------------------------------------------------------------------
def plot_pulse(model, tokenizer, text, device, layer_idx=-1):
    model.eval()
    tokens = tokenizer.encode(text).ids
    temps_list, amn_list, words = [], [], []
    temps = [torch.ones(1,1,device=device) for _ in range(model.num_layers)]
    amnes = [torch.zeros(1,1,device=device) for _ in range(model.num_layers)]
    with torch.no_grad():
        for i in range(1, len(tokens)):
            x = torch.tensor([tokens[:i]], dtype=torch.long).to(device)
            out = model.embedding(x)
            for j, layer in enumerate(model.layers):
                out, temps[j], amnes[j] = layer(out, temps[j], amnes[j])
            w = tokenizer.decode([tokens[i]]).replace('Ġ',' ').replace('Ċ','\n').strip()
            if not w: w = ' '
            words.append(w)
            temps_list.append(temps[layer_idx].item())
            amn_list.append(amnes[layer_idx].item())
    ct = model.layers[layer_idx].pulse.critical_temp
    fig, (ax1, ax2) = plt.subplots(2,1,figsize=(14,9))
    ax1.plot(temps_list, marker='o', color='crimson', linewidth=2.5, markersize=8)
    ax1.axhline(y=ct, color='gray', linestyle='--', linewidth=2, label=f'Critical ({ct})')
    ax1.set_xticks(range(len(words))); ax1.set_xticklabels(words, rotation=45, ha='right', fontsize=10)
    ax1.set_ylabel('Temperature'); ax1.set_title(f'PulseTransformer Heartbeat — Layer {layer_idx}')
    ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.plot(amn_list, marker='s', color='darkblue', linewidth=2.5, markersize=7)
    ax2.set_xticks(range(len(words))); ax2.set_xticklabels(words, rotation=45, ha='right', fontsize=10)
    ax2.set_ylabel('Amnesia'); ax2.set_title('Amnesia Accumulation')
    ax2.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig('pulse_heartbeat.png', dpi=150); plt.show()
    print(f"\nStats: mean T={np.mean(temps_list):.3f}, max T={np.max(temps_list):.3f}, peaks={sum(1 for t in temps_list if t>ct)}")

print("\nЗапись пульса...")
plot_pulse(pulse, tokenizer, "Once upon a time, a little girl named Lily went to the forest. She saw a magic flower and smiled.", device)

print("\n" + "="*50)
print("ГОТОВО! Таблица и график пульса получены.")
print("="*50)
