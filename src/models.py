import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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
        B, L, _ = x.shape
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1,2)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1,2)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1,2)
        cos_cur = self.cos[:L].unsqueeze(0).unsqueeze(0)
        sin_cur = self.sin[:L].unsqueeze(0).unsqueeze(0)
        q, k = apply_rotary_pos_emb(q, k, cos_cur, sin_cur)
        scale = math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2,-1)) / scale
        if causal_mask:
            mask = torch.triu(torch.ones(L, L, device=x.device)*float('-inf'), diagonal=1)
            scores = scores + mask
        attn_probs = F.softmax(scores, dim=-1)
        attn_probs = F.dropout(attn_probs, p=self.dropout, training=self.training)
        context = torch.matmul(attn_probs, v)
        context = context.transpose(1,2).contiguous().view(B, L, self.embed_dim)
        return self.out_proj(context)

class HomeostaticModule(nn.Module):
    def __init__(self, embed_dim, critical_temp=5.0):
        super().__init__()
        self.critical_temp = critical_temp
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.temp_gate = nn.Linear(embed_dim, 1)
        self.temp_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x, prev_temp, prev_amnesia):
        q = self.query_proj(x[:, -1, :])
        k = self.key_proj(x).mean(dim=1)
        info_density = torch.sigmoid((q * k).sum(dim=-1, keepdim=True))
        delta_temp = torch.tanh(self.temp_gate(x[:, -1, :]))
        new_temp = prev_temp + delta_temp * info_density
        new_temp = torch.clamp(new_temp, min=0.2, max=6.0)
        amnesia_gate = torch.sigmoid(new_temp - self.critical_temp)
        new_amnesia = prev_amnesia * (1 - amnesia_gate) + amnesia_gate
        modulation = 1.0 / (1.0 + self.temp_scale * new_temp.unsqueeze(1))
        x_modulated = x * modulation
        return x_modulated, new_temp, new_amnesia

class BaselineDecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1, max_seq_len=512):
        super().__init__()
        self.attn = RotaryMultiheadAttention(embed_dim, num_heads, dropout, max_seq_len)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out = self.attn(self.norm1(x), causal_mask=True)
        x = x + self.dropout(attn_out)
        ff_out = self.ff(self.norm2(x))
        x = x + self.dropout(ff_out)
        return x

class BaselineTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_layers=3, num_heads=4, ff_dim=256, max_seq_len=512):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.layers = nn.ModuleList([
            BaselineDecoderLayer(embed_dim, num_heads, ff_dim, max_seq_len=max_seq_len)
            for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, x, return_hidden=False):
        out = self.embedding(x)
        for layer in self.layers:
            out = layer(out)
        logits = self.fc_out(out)
        if return_hidden:
            return logits, out
        return logits

class HomeostaticDecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1, max_seq_len=512, critical_temp=5.0):
        super().__init__()
        self.attn = RotaryMultiheadAttention(embed_dim, num_heads, dropout, max_seq_len)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.homeo = HomeostaticModule(embed_dim, critical_temp=critical_temp)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, prev_temp, prev_amnesia):
        attn_out = self.attn(self.norm1(x), causal_mask=True)
        x = x + self.dropout(attn_out)
        ff_out = self.ff(self.norm2(x))
        x = x + self.dropout(ff_out)
        x_mod, new_temp, new_amnesia = self.homeo(x, prev_temp, prev_amnesia)
        return x_mod, new_temp, new_amnesia

class HomeostaticTransformerV2(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_layers=3, num_heads=4, ff_dim=256, max_seq_len=512, critical_temp=5.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.layers = nn.ModuleList([
            HomeostaticDecoderLayer(embed_dim, num_heads, ff_dim, max_seq_len=max_seq_len, critical_temp=critical_temp)
            for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, x, temps=None, amnesias=None, return_hidden=False):
        B = x.size(0)
        if temps is None:
            temps = [torch.ones(B,1,device=x.device) for _ in range(self.num_layers)]
        if amnesias is None:
            amnesias = [torch.zeros(B,1,device=x.device) for _ in range(self.num_layers)]
        out = self.embedding(x)
        new_temps, new_amnesias = [], []
        for layer, t, a in zip(self.layers, temps, amnesias):
            out, t_new, a_new = layer(out, t, a)
            new_temps.append(t_new)
            new_amnesias.append(a_new)
        logits = self.fc_out(out)
        if return_hidden:
            return logits, new_temps, new_amnesias, out
        return logits, new_temps, new_amnesias
