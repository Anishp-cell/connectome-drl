"""Connectome-Constrained Policy Architecture (Model 1 from Table: MaskedLinear).

A bio-constrained Actor-Critic policy where the internal motor pathways strictly
adhere to the Janelia MaleCNS v1.0 connectome wiring diagram:
  Sensory Inputs -> 4 DNs -> 125 VNC Interneurons -> 377 Motor Neurons -> 42 Joints
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

from connectome_rl.src.models.connectome_layers import MaskedLinear
from connectome_rl.src.models.mlp_policy import layer_init
from connectome_rl.src.connectome.graph_utils import ConnectomeCircuitData, load_circuit_data


class ConnectomePolicy(nn.Module):
    """Actor-Critic Policy constrained by the MaleCNS fruit fly connectome.
    
    Motor Hierarchy:
      1. Sensory Encoder: Maps high-dimensional observations (e.g. 100-dim) to
         the 4 Descending Neurons (DNa01/DNa02 steering & speed command channels).
      2. Hop 1 MaskedLinear: Biological connections from 4 DNs to 125 VNC Interneurons.
      3. Hop 2 MaskedLinear: Biological connections from 125 VNC Interneurons to 377 Motor Neurons.
      4. Actuator Decoder: Linear mapping from 377 motor neurons to 42 leg joint targets.
      5. Critic: Parallel value network estimating state-value V(s).
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        circuit_data: ConnectomeCircuitData | str | Path,
        activation: str = "tanh",
        init_log_std: float = -0.5,
        use_biological_weights: bool = True,
    ) -> None:
        """Initialize the connectome-constrained policy.
        
        Args:
            obs_dim: Dimension of observation vector.
            act_dim: Dimension of action vector (42 for FlyGym).
            circuit_data: ConnectomeCircuitData object or path to .pt file.
            activation: Activation function ('tanh' or 'relu').
            init_log_std: Initial exploration log standard deviation.
            use_biological_weights: If True, initialize weights using biological synapse counts.
        """
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        # Load circuit data if a path is provided
        if isinstance(circuit_data, (str, Path)):
            circuit_data = load_circuit_data(circuit_data)
        self.circuit_data = circuit_data

        act_fn = nn.Tanh if activation.lower() == "tanh" else nn.ReLU

        num_dn = len(circuit_data.dn_indices)
        num_in = len(circuit_data.in_indices)
        num_mn = len(circuit_data.mn_indices)

        # 1. Sensory Encoder: maps raw sensory features -> 4 Descending Neurons
        self.sensory_encoder = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            act_fn(),
            layer_init(nn.Linear(64, num_dn)),
            act_fn(),
        )

        # 2. Hop 1 Layer: 4 DNs -> 125 VNC Interneurons (MaskedLinear)
        init_hop1_w = circuit_data.hop1_weights if use_biological_weights else None
        self.hop1_layer = MaskedLinear(
            in_features=num_dn,
            out_features=num_in,
            mask=circuit_data.hop1_mask,
            initial_weights=init_hop1_w,
        )
        self.act1 = act_fn()

        # 3. Hop 2 Layer: 125 VNC Interneurons -> 377 Motor Neurons (MaskedLinear)
        init_hop2_w = circuit_data.hop2_weights if use_biological_weights else None
        self.hop2_layer = MaskedLinear(
            in_features=num_in,
            out_features=num_mn,
            mask=circuit_data.hop2_mask,
            initial_weights=init_hop2_w,
        )
        self.act2 = act_fn()

        # 4. Actuator Decoder: maps 377 Motor Neurons -> 42 leg joint actions
        self.actuator_decoder = layer_init(nn.Linear(num_mn, act_dim), std=0.01)

        # Learnable Gaussian exploration parameter
        self.actor_logstd = nn.Parameter(torch.ones(act_dim) * init_log_std)

        # 5. Critic (Value network)
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 128)),
            act_fn(),
            layer_init(nn.Linear(128, 128)),
            act_fn(),
            layer_init(nn.Linear(128, 1), std=1.0),
        )

    def forward_actor(self, obs: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Run forward pass through the biological motor pathway.
        
        Returns:
            action_mean: Tensor of shape (batch_size, 42).
            activations: Dictionary storing intermediate biological firings
                        ('dn', 'interneuron', 'motor_neuron') for analysis.
        """
        # Step 1: Sensory input drives the 4 Descending Neurons
        dn_activity = self.sensory_encoder(obs)
        
        # Step 2: DNs activate VNC Interneurons via biological synapses
        in_activity = self.act1(self.hop1_layer(dn_activity))
        
        # Step 3: Interneurons activate Motor Neurons via biological synapses
        mn_activity = self.act2(self.hop2_layer(in_activity))
        
        # Step 4: Motor neurons pull the 42 leg joints
        action_mean = self.actuator_decoder(mn_activity)

        activations = {
            "dn": dn_activity,
            "interneuron": in_activity,
            "motor_neuron": mn_activity,
        }
        return action_mean, activations

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Estimate state value V(s)."""
        return self.critic(obs)

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute action distribution, sample action (or evaluate given action), and state value."""
        action_mean, _ = self.forward_actor(obs)
        action_logstd = torch.clamp(self.actor_logstd, min=-20.0, max=2.0)
        action_std = torch.exp(action_logstd)

        dist = Normal(action_mean, action_std)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action).sum(axis=-1)
        entropy = dist.entropy().sum(axis=-1)
        value = self.critic(obs)

        return action, log_prob, entropy, value

    def get_deterministic_action(self, obs: torch.Tensor) -> torch.Tensor:
        """Get deterministic mean action for evaluation and rendering."""
        with torch.no_grad():
            action_mean, _ = self.forward_actor(obs)
            return action_mean

    # --- In-Silico Lesion Controls ---

    def lesion_dn_channel(self, dn_channel: int) -> int:
        """Silence a specific Descending Neuron (e.g. 0 or 1 for DNa01 steering)."""
        return self.hop1_layer.apply_lesion(target_cols=[dn_channel])

    def lesion_motor_neurons(self, mn_indices: list[int]) -> int:
        """Silence specific Motor Neurons to simulate peripheral nerve damage."""
        return self.hop2_layer.apply_lesion(target_rows=mn_indices)

    def reset_lesions(self) -> None:
        """Restore all intact biological connections."""
        self.hop1_layer.reset_lesions()
        self.hop2_layer.reset_lesions()


if __name__ == "__main__":
    # Smoke test: load real cached circuit and test forward pass
    torch.manual_seed(42)
    data_dir = Path(__file__).resolve().parents[2] / "data"
    circuit_path = data_dir / "dna_circuit_tensors.pt"
    
    if not circuit_path.exists():
        print(f"Error: {circuit_path} not found. Run graph_utils.py first!")
        exit(1)
        
    batch_size = 4
    dummy_obs_dim = 100
    dummy_act_dim = 42
    
    policy = ConnectomePolicy(
        obs_dim=dummy_obs_dim,
        act_dim=dummy_act_dim,
        circuit_data=circuit_path,
        use_biological_weights=True,
    )
    
    print("=" * 65)
    print("CONNECTOME POLICY (MASKED LINEAR MODEL) SMOKE TEST")
    print("=" * 65)
    print(f"Hop 1 Layer: {policy.hop1_layer}")
    print(f"Hop 2 Layer: {policy.hop2_layer}")
    
    dummy_obs = torch.randn(batch_size, dummy_obs_dim)
    action, log_prob, entropy, value = policy.get_action_and_value(dummy_obs)
    
    print(f"\nForward output action shape: {action.shape}")
    print(f"State value V(s) shape:      {value.shape}")
    
    # Backward pass test
    loss = value.mean() - log_prob.mean()
    loss.backward()
    
    # Check that masked gradients are strictly 0.0 in both biological layers
    hop1_masked_grad = policy.hop1_layer.weight.grad[~policy.hop1_layer.mask]
    hop2_masked_grad = policy.hop2_layer.weight.grad[~policy.hop2_layer.mask]
    
    assert (hop1_masked_grad == 0.0).all(), "Hop 1 masked gradients must be strictly 0.0!"
    assert (hop2_masked_grad == 0.0).all(), "Hop 2 masked gradients must be strictly 0.0!"
    print("✅ Biological gradient isolation verified in both connectome layers!")
    
    # Lesion test
    print("\n--- In-Silico Lesion Test (Silencing DNa01) ---")
    silenced = policy.lesion_dn_channel(0)
    print(f"Silenced {silenced} synapses leaving DNa01.")
    
    # Forward pass under lesion
    lesioned_action = policy.get_deterministic_action(dummy_obs)
    policy.reset_lesions()
    intact_action = policy.get_deterministic_action(dummy_obs)
    diff = (lesioned_action - intact_action).abs().mean().item()
    print(f"Mean motor output difference under DNa01 lesion: {diff:.4f}")
    assert diff > 0.0, "Lesion must alter downstream motor output!"
    print("✅ Lesion successfully altered motor commands!")
    print("=" * 65)
