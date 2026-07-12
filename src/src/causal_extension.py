# =============================================================================
# CAUSAL EXTENSION WITH ANTI-FREEZE STABILIZER (causal_extension.py)
# Финальная производственная версия. Исправляет замерзание температуры.
# Поместить в папку: src/src/ рядом с оригинальным experiment_runner.py
# =============================================================================

import torch
import torch.nn as nn
import math

try:
    from experiment_runner import RotaryMultiheadAttention, PulseModule
except ImportError:
    pass

class CausalPulseModule(nn.Module):
    def __init__(self, embed_dim, critical_temp=2.0, temp_scale=0.3):
        super().__init__()
        self.critical_temp = critical_temp
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.temp_gate = nn.Linear(embed_dim, 1)
        self.temp_scale = nn.Parameter(torch.tensor(temp_scale))
        nn.init.xavier_uniform_(self.temp_gate.weight, gain=1.5)

    def forward(self, x, prev_temp, prev_amnesia):
        """
        Честный параллельный каузальный расчет с антифриз-стабилизатором Ле Шателье.
        """
        B, S, _ = x.shape
        
        q = self.query_proj(x)  # [B, S, embed_dim]
        k = self.key_proj(x)    # [B, S, embed_dim]
        
        info_density = torch.sigmoid((q * k).sum(dim=-1, keepdim=True))  # [B, S, 1]
        delta_temp = torch.tanh(self.temp_gate(x)) * 0.5  # [B, S, 1]
        step_updates = delta_temp * (info_density + 0.1)  # [B, S, 1]
        
        if self.training:
            current_temp = prev_temp.unsqueeze(1)
            cumsum_updates = []
            
            for t in range(S):
                # Сила пассивного возврата: удерживает систему от замерзания в 0.1
                passive_return = (1.0 - current_temp) * 0.15 
                step_mod = step_updates[:, t:t+1, :] + passive_return
                current_temp = torch.clamp(current_temp + step_mod, min=0.1, max=5.0)
                cumsum_updates.append(current_temp)
                
            new_temp = torch.cat(cumsum_updates, dim=1)
            amnesia_gate = torch.sigmoid(new_temp - self.critical_temp)
            decay_factor = torch.cumprod(1 - amnesia_gate, dim=1)
            
            amnesia_from_gates = amnesia_gate / torch.clamp(decay_factor, min=1e-6)
            cumsum_amnesia = torch.cumsum(amnesia_from_gates * decay_factor, dim=1)
            new_amnesia = prev_amnesia.unsqueeze(1) * decay_factor + cumsum_amnesia * decay_factor
            new_amnesia = torch.clamp(new_amnesia, min=0.0, max=1.0)
            
            final_temp = new_temp[:, -1, :]
            final_amnesia = new_amnesia[:, -1, :]
        else:
            # Режим инференса и отрисовки графика Heartbeat
            passive_return = (1.0 - prev_temp.unsqueeze(1)) * 0.15
            new_temp = prev_temp + step_updates[:, -1, :] + passive_return[:, -1, :]
            new_temp = torch.clamp(new_temp, min=0.1, max=5.0)
            
            amnesia_gate = torch.sigmoid(new_temp - self.critical_temp)
            new_amnesia = prev_amnesia * (1 - amnesia_gate) + amnesia_gate
            
            new_temp = new_temp.unsqueeze(1)
            new_amnesia = new_amnesia.unsqueeze(1)
            final_temp = new_temp[:, -1, :]
            final_amnesia = new_amnesia[:, -1, :]

        modulation = torch.exp(-new_temp / 10.0)
        return x * modulation, final_temp, final_amnesia

class CausalPulseTransformerLayer(nn.Module):
    def __init__(self, original_layer: nn.Module):
        super().__init__()
        self.attention = original_layer.attention
        self.ff = original_layer.ff
        self.norm1 = original_layer.norm1
        self.norm2 = original_layer.norm2
        self.dropout = original_layer.dropout
        
        self.pulse = CausalPulseModule(
            embed_dim=original_layer.pulse.query_proj.in_features,
            critical_temp=original_layer.pulse.critical_temp
        )

    def forward(self, x, prev_temp, prev_amnesia):
        attn_out = self.attention(x, causal_mask=True)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return self.pulse(x, prev_temp, prev_amnesia)

class CausalPulseTransformer(nn.Module):
    def __init__(self, original_model: nn.Module):
        super().__init__()
        self.embedding = original_model.embedding
        self.fc_out = original_model.fc_out
        self.num_layers = original_model.num_layers
        self.layers = nn.ModuleList([
            CausalPulseTransformerLayer(layer) for layer in original_model.layers
        ])

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
