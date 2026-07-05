# =============================================================================
# ГОМЕОСТАТИЧЕСКИЙ ТРАНСФОРМЕР С ФАЗАМИ СНА – ИСПРАВЛЕННАЯ ВЕРСИЯ
# =============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
import matplotlib.pyplot as plt
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Устройство: {device}")

# -----------------------------------------------------------------------------
# 1. RoPE + RotaryMultiheadAttention
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

# -----------------------------------------------------------------------------
# 2. Гомеостатический модуль (ИСПРАВЛЕННАЯ версия)
# -----------------------------------------------------------------------------
class HomeostaticModule(nn.Module):
    def __init__(self, embed_dim, critical_temp=4.0, temp_scale=0.3):
        super().__init__()
        self.critical_temp = critical_temp
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.temp_gate = nn.Linear(embed_dim, 1)
        self.temp_scale = nn.Parameter(torch.tensor(temp_scale))

    def forward(self, x, prev_temp, prev_amnesia):
        q = self.query_proj(x[:, -1, :])
        k = self.key_proj(x).mean(dim=1)
        info_density = torch.sigmoid((q * k).sum(dim=-1, keepdim=True))

        delta_temp = torch.tanh(self.temp_gate(x[:, -1, :]))
        # Небольшой положительный дрейф, чтобы температура не застревала на 1.0
        delta_temp = delta_temp + 0.05
        new_temp = prev_temp + delta_temp * info_density
        new_temp = torch.clamp(new_temp, min=0.5, max=5.0)

        amnesia_gate = torch.sigmoid(new_temp - self.critical_temp)
        new_amnesia = prev_amnesia * (1 - amnesia_gate) + amnesia_gate

        modulation = 1.0 / (1.0 + self.temp_scale * new_temp.unsqueeze(1))
        x_modulated = x * modulation

        return x_modulated, new_temp, new_amnesia

# -----------------------------------------------------------------------------
# 3. Трансформерный слой с гомеостазом
# -----------------------------------------------------------------------------
class HomeostaticTransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_heads=4, ff_dim=256, critical_temp=4.0, temp_scale=0.3, dropout=0.1, max_seq_len=512):
        super().__init__()
        self.attention = RotaryMultiheadAttention(embed_dim, num_heads, dropout, max_seq_len)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.homeo = HomeostaticModule(embed_dim, critical_temp, temp_scale)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, prev_temp, prev_amnesia):
        attn_out = self.attention(x, causal_mask=True)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        x_mod, new_temp, new_amnesia = self.homeo(x, prev_temp, prev_amnesia)
        return x_mod, new_temp, new_amnesia

# -----------------------------------------------------------------------------
# 4. Полная модель
# -----------------------------------------------------------------------------
class HomeostaticTransformerV2(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_layers=3, num_heads=4, ff_dim=256,
                 critical_temp=4.0, temp_scale=0.3, max_seq_len=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.layers = nn.ModuleList([
            HomeostaticTransformerLayer(embed_dim, num_heads, ff_dim, critical_temp, temp_scale, max_seq_len=max_seq_len)
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
        logits = self.fc_out(out)
        return logits, new_temps, new_amnesias

# -----------------------------------------------------------------------------
# 5. Данные и токенизатор
# -----------------------------------------------------------------------------
print("Загрузка данных TinyStories и обучение BPE...")
dataset = load_dataset("roneneldan/TinyStories", split="train[:10000]")

tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
trainer = trainers.BpeTrainer(vocab_size=5000, special_tokens=["<pad>", "<unk>", "<s>", "</s>"])
tokenizer.train_from_iterator(dataset["text"], trainer)
vocab_size = tokenizer.get_vocab_size()

MAX_LEN = 64
full_dataset = load_dataset("roneneldan/TinyStories", split="train[:30000]")
tokenized = [tokenizer.encode(t).ids for t in full_dataset["text"]]
tokenized = [t[:MAX_LEN] for t in tokenized if len(t) > 5]
padded = [t + [0]*(MAX_LEN - len(t)) for t in tokenized]

X = torch.tensor([t[:-1] for t in padded], dtype=torch.long).to(device)
Y = torch.tensor([t[1:] for t in padded], dtype=torch.long).to(device)

# -----------------------------------------------------------------------------
# 6. Обучение (5 эпох)
# -----------------------------------------------------------------------------
model = HomeostaticTransformerV2(vocab_size=vocab_size, critical_temp=4.0, temp_scale=0.3).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)

epochs = 5
batch_size = 128
print(f"Обучение на {epochs} эпохах (30k примеров)...")
for epoch in range(epochs):
    model.train()
    total_loss, total_amnesia = 0.0, 0.0
    perm = torch.randperm(X.size(0))
    for i in range(0, X.size(0), batch_size):
        idx = perm[i:i+batch_size]
        bx, by = X[idx], Y[idx]
        optimizer.zero_grad()
        logits, temps, amnesias = model(bx)
        loss = criterion(logits.view(-1, vocab_size), by.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_amnesia += sum(a.mean().item() for a in amnesias)
    print(f"Эпоха {epoch+1}: loss={total_loss/(i//batch_size+1):.3f}, "
          f"T={temps[-1].mean().item():.2f}, Am={total_amnesia/(i//batch_size+1):.3f}")

# -----------------------------------------------------------------------------
# 7. Генерация с активным «сном»
# -----------------------------------------------------------------------------
print("\nГенерация 3 историй с фазами бодрствования и сна...")
model.eval()

prompts = [
    "Once upon a time",
    "In a beautiful garden",
    "Suddenly, a little bird"
]

unk_id = tokenizer.token_to_id("<unk>")
pad_id = tokenizer.token_to_id("<pad>")
sleep_steps = 15
gen_steps = 60

global_temps = [[] for _ in range(model.num_layers)]
global_amnesia = [[] for _ in range(model.num_layers)]

for i, start_text in enumerate(prompts):
    print(f"\n--- ИСТОРИЯ {i+1}: '{start_text}' ---")
    start_ids = tokenizer.encode(start_text).ids
    generated = torch.tensor([start_ids], dtype=torch.long, device=device)

    temps = [torch.ones(1, 1, device=device) for _ in range(model.num_layers)]
    amnesias = [torch.zeros(1, 1, device=device) for _ in range(model.num_layers)]

    # Бодрствование
    with torch.no_grad():
        for step in range(gen_steps):
            logits, temps, amnesias = model(generated, temps, amnesias)

            for l in range(model.num_layers):
                global_temps[l].append(temps[l].mean().item())
                global_amnesia[l].append(amnesias[l].mean().item())

            next_logits = logits[0, -1, :].clone()
            next_logits[unk_id] = float('-inf')
            next_logits[pad_id] = float('-inf')
            probs = F.softmax(next_logits / 0.9, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).unsqueeze(0)
            generated = torch.cat([generated, next_token], dim=1)
            if generated.size(1) > MAX_LEN:
                generated = generated[:, -MAX_LEN:]

    decoded = tokenizer.decode(generated[0].tolist())
    decoded = decoded.replace('Ġ', ' ').replace('Ċ', '\n').strip()
    print("Результат бодрствования:")
    print(decoded[:500])

    # Сон
    print(f"Сон ({sleep_steps} шагов)...")
    dummy_ids = torch.zeros(1, MAX_LEN, dtype=torch.long, device=device)
    for _ in range(sleep_steps):
        with torch.no_grad():
            _, temps, amnesias = model(dummy_ids, temps, amnesias)
            for l in range(model.num_layers):
                global_temps[l].append(temps[l].mean().item())
                global_amnesia[l].append(amnesias[l].mean().item())
    # Глубокий сброс
    temps = [torch.ones(1, 1, device=device) for _ in range(model.num_layers)]
    amnesias = [torch.zeros(1, 1, device=device) for _ in range(model.num_layers)]
    print("Сон завершён.\n")

# -----------------------------------------------------------------------------
# 8. Визуализация
# -----------------------------------------------------------------------------
plt.figure(figsize=(14, 6))
plt.subplot(2, 1, 1)
for l in range(model.num_layers):
    plt.plot(global_temps[l], label=f'Слой {l}')
plt.axhline(y=4.0, color='red', linestyle='--', label='critical_temp=4.0')
plt.title('Температура (гомеостатическая)')
plt.ylabel('Температура')
plt.legend()

plt.subplot(2, 1, 2)
for l in range(model.num_layers):
    plt.plot(global_amnesia[l], label=f'Слой {l}')
plt.title('Амнезия')
plt.xlabel('Суммарные шаги')
plt.ylabel('Уровень амнезии')
plt.legend()
plt.tight_layout()
plt.show()
https://colab.research.google.com/drive/1eEEd67PpfQh75idfQ3https://colab.research.google.com/drive/1eEEd67PpfQh75idfQ3vY5_tcHPpHyOW2#scrollTo=-8RBj176a32ivY5_tcHPpHyOW2#scrollTo=-8RBj176a32i
