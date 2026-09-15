"""Connectome-Constrained Neural Network Layers.

Implements the MaskedLinear layer: a PyTorch linear module whose connectivity
is strictly constrained by a biological connectome adjacency mask. Non-existent
synapses are permanently clamped to zero in both forward activations and
backward gradient propagation.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedLinear(nn.Module):
    """Linear layer constrained by a biological connectome connectivity mask.
    
    Forward equation:
        Y = X (W ⊙ M_bio ⊙ M_lesion)^T + bias
        
    Where:
      - W is the learnable weight matrix of shape (out_features, in_features).
      - M_bio is a fixed boolean buffer (True = biological synapse exists).
      - M_lesion is a dynamic boolean buffer for in-silico ablation experiments.
      - ⊙ is element-wise multiplication.
      
    Gradient Guarantee:
      A backward hook forces gradients at non-existent synapses to be strictly 0.0,
      ensuring that un-wired connections can NEVER be learned by the optimizer.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        mask: torch.BoolTensor,
        initial_weights: torch.FloatTensor | None = None,
        bias: bool = True,
        weight_scale: float = 0.1,
    ) -> None:
        """Initialize MaskedLinear layer.
        
        Args:
            in_features: Number of input neurons (upstream sources).
            out_features: Number of output neurons (downstream targets).
            mask: Boolean tensor of shape (out_features, in_features).
                  mask[i, j] == True means source j connects to target i.
            initial_weights: Optional float tensor of biological synapse weights.
            bias: Whether to include an additive bias vector.
            weight_scale: Initial random scaling factor for active synapses.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        if mask.shape != (out_features, in_features):
            raise ValueError(
                f"Mask shape {mask.shape} does not match (out_features={out_features}, "
                f"in_features={in_features})"
            )

        # Register biological mask as a persistent non-parameter buffer
        self.register_buffer("mask", mask.bool())
        
        # Dynamic lesion mask (all True by default; zeroed out during lesion experiments)
        self.register_buffer("lesion_mask", torch.ones_like(mask, dtype=torch.bool))

        # Learnable weight parameter
        self.weight = nn.Parameter(torch.empty(out_features, in_features))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        # Initialize weights
        self._init_weights(initial_weights, weight_scale)

        # Attach backward hook: forces gradients at zeroed positions to stay 0.0
        self.weight.register_hook(self._mask_gradient_hook)

    def _init_weights(
        self,
        initial_weights: torch.FloatTensor | None,
        weight_scale: float,
    ) -> None:
        """Initialize weights only at active biological synapses."""
        with torch.no_grad():
            if initial_weights is not None:
                # Use biological synapse strengths as initial weights
                # Randomize signs (+ for excitation, - for inhibition)
                signs = torch.randint(0, 2, self.weight.shape).float() * 2.0 - 1.0
                init_w = initial_weights.float() * signs * weight_scale
                self.weight.copy_(init_w * self.mask)
            else:
                # Orthogonal-style initialization scaled by active fan-in
                active_per_row = self.mask.sum(dim=1, keepdim=True).float().clamp(min=1.0)
                std = weight_scale / torch.sqrt(active_per_row)
                self.weight.normal_(0.0, 1.0)
                self.weight.mul_(std)
                self.weight.mul_(self.mask)

    def _mask_gradient_hook(self, grad: torch.Tensor) -> torch.Tensor:
        """Backward hook that zeroes out gradients where synapses do not exist."""
        effective_mask = self.mask & self.lesion_mask
        return grad * effective_mask

    @property
    def effective_mask(self) -> torch.Tensor:
        """Combines the static biological mask and the dynamic lesion mask."""
        return self.mask & self.lesion_mask

    @property
    def effective_weight(self) -> torch.Tensor:
        """Returns the actual weights used in computation: W ⊙ M_bio ⊙ M_lesion."""
        return self.weight * self.effective_mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: Y = X @ effective_W^T + bias."""
        w = self.effective_weight
        return F.linear(x, w, self.bias)

    # --- In-Silico Lesion Experiment Controls ---

    def apply_lesion(
        self,
        target_rows: list[int] | None = None,
        target_cols: list[int] | None = None,
    ) -> int:
        """Ablate (zero out) specific biological neurons in real time.
        
        Args:
            target_rows: Downstream neuron indices whose inputs should be silenced.
            target_cols: Upstream neuron indices whose outputs should be silenced.
            
        Returns:
            Number of newly silenced synaptic connections.
        """
        before_active = self.effective_mask.sum().item()
        
        if target_rows is not None and len(target_rows) > 0:
            self.lesion_mask[target_rows, :] = False
            
        if target_cols is not None and len(target_cols) > 0:
            self.lesion_mask[:, target_cols] = False

        after_active = self.effective_mask.sum().item()
        silenced = before_active - after_active
        return silenced

    def reset_lesions(self) -> None:
        """Restore the intact biological mask, reversing all in-silico lesions."""
        self.lesion_mask.fill_(True)

    def extra_repr(self) -> str:
        active_count = self.mask.sum().item()
        total_count = self.mask.numel()
        density = (active_count / total_count) * 100.0 if total_count > 0 else 0.0
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"active_synapses={active_count}/{total_count} ({density:.2f}% density), "
            f"bias={self.bias is not None}"
        )


if __name__ == "__main__":
    # Smoke test: create a toy MaskedLinear layer and verify gradients & lesions
    torch.manual_seed(42)
    in_dim = 4    # e.g. 4 Descending Neurons
    out_dim = 6   # e.g. 6 Interneurons
    
    # Toy sparse mask (only 5 out of 24 connections exist)
    toy_mask = torch.tensor([
        [True,  False, False, False],
        [True,  True,  False, False],
        [False, True,  False, False],
        [False, False, True,  False],
        [False, False, True,  True ],
        [False, False, False, True ],
    ], dtype=torch.bool)
    
    layer = MaskedLinear(in_features=in_dim, out_features=out_dim, mask=toy_mask)
    print("=" * 60)
    print("MASKED LINEAR LAYER SMOKE TEST")
    print("=" * 60)
    print(f"Layer: {layer}")
    
    # Test 1: Forward pass
    dummy_input = torch.ones(2, in_dim)
    output = layer(dummy_input)
    print(f"Forward output shape: {output.shape}")
    
    # Test 2: Verify non-existent weights are strictly 0.0
    eff_w = layer.effective_weight
    assert (eff_w[~toy_mask] == 0.0).all(), "Non-existent weights must be 0.0!"
    print("✅ Forward pass correctly zeroes un-wired connections.")
    
    # Test 3: Backward pass & gradient isolation
    loss = output.sum()
    loss.backward()
    grad = layer.weight.grad
    assert (grad[~toy_mask] == 0.0).all(), "Gradients at masked positions must be 0.0!"
    assert (grad[toy_mask] != 0.0).all(), "Gradients at active positions must be non-zero!"
    print("✅ Gradient backward hook strictly isolates non-existent connections.")
    
    # Test 4: In-silico Lesion test
    print("\n--- Lesion Experiment Test ---")
    active_before = layer.effective_mask.sum().item()
    print(f"Active synapses before lesion: {active_before}")
    
    # Cut output neuron #1
    silenced = layer.apply_lesion(target_rows=[1])
    active_after = layer.effective_mask.sum().item()
    print(f"Silenced {silenced} synapses. Active synapses remaining: {active_after}")
    assert active_after < active_before
    
    # Reset lesion
    layer.reset_lesions()
    assert layer.effective_mask.sum().item() == active_before
    print("✅ In-silico lesioning and reset verified!")
    print("=" * 60)
