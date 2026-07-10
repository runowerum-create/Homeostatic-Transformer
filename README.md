 .mean().item() calls)
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
