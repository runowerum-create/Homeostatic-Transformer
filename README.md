import torch
import torch.nn as nn
import torch.nn.functional as F

class HomeostaticVortexLayer(nn.Module):
    """
    Implementation of the Eulerian Homeostatic Layer 
    with a closed-loop feedback mechanism, optimized for PyTorch gradients.
    """
    def __init__(self, d_model, target_temp=0.3, lambda_param=0.1, critical_temp=2.0):
        super().__init__()
        self.d_model = d_model
        
        # Homeostatic parameters
        self.target_temp = target_temp      # tau_target (optimal entropy balance)
        self.lambda_param = lambda_param    # System thermal capacity parameter
        self.critical_temp = critical_temp  # Threshold for protective inhibition
        
        # Projections to analyze informational pressure (Eulerian flow analogue)
        self.query_flow = nn.Linear(d_model, d_model)
        self.context_flow = nn.Linear(d_model, d_model)
        
        # Projections for memory vortex transformation
        self.vortex_update = nn.Linear(d_model, d_model)
        
    def forward(self, x, memory_vortex=None):
        """
        x: [batch_size, seq_len, d_model] - current batch hidden states
        memory_vortex: [batch_size, d_model] - current state of the environmental memory vortex
        """
        batch_size, seq_len, d_model = x.shape
        
        # Initialize vortex memory if it doesn't exist (start of the flow)
        if memory_vortex is None:
            memory_vortex = torch.zeros(batch_size, d_model, device=x.device)
            
        # 1. Determine local informational pressure (Dynamic Tau)
        # Compare each token with the mean context of the current batch
        q = self.query_flow(x)                       # [batch_size, seq_len, d_model]
        k = self.context_flow(x).mean(dim=1, keepdim=True) # [batch_size, 1, d_model]
        
        # Calculate tau individually for each token in the batch using cosine similarity
        # tau: [batch_size, seq_len]
        tau = 1.0 - F.cosine_similarity(q, k, dim=-1)
        
        # 2. Implement the NEW homeostatic feedback equation:
        # x * exp((tau_target - tau) / lambda)
        # Unsqueeze to [batch_size, seq_len, 1] for proper tensor broadcasting
        homeostatic_scale = torch.exp((self.target_temp - tau) / self.lambda_param).unsqueeze(-1)
        x_modulated = x * homeostatic_scale
        
        # 3. Differentiable Amnesia Gate (replaces broken .mean().item() calls)
        # Smoothly activates from 0 to 1 when local tau exceeds critical_temp
        # A pure sigmoid function preserves the PyTorch computational graph
        amnesia_gate = torch.sigmoid((tau - self.critical_temp) / 0.1).mean(dim=1, keepdim=True) # [batch_size, 1]
        
        # 4. Eulerian Memory Vortex update
        # The current token flow in the batch shapes the new environmental swirl
        current_vortex_contribution = self.vortex_update(x_modulated).mean(dim=1) # [batch_size, d_model]
        
        # Apply environmental viscosity and total wipeout during protective inhibition
        # (1.0 - amnesia_gate) collapses the vortex under high entropy stress
        memory_vortex = memory_vortex * (1.0 - amnesia_gate) + current_vortex_contribution * (1.0 - amnesia_gate)
        
        # Inject vortex memory dynamics back into the modulated hidden states flow
        x_final = x_modulated + memory_vortex.unsqueeze(1)
        
        return x_final, memory_vortex
