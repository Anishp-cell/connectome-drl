"""Hierarchical Connectome Policy Architecture (Model 4: Two-Tier Brain -> Low-Level VNC).

Biologically grounded two-tier motor hierarchy:
  1. High-Level Cephalic Brain:
     Processes high-dimensional sensory inputs and goals, producing a compact 4-dimensional
     descending command vector z = [DNa01_L, DNa01_R, DNa02_L, DNa02_R] representing
     high-level steering and locomotion speed intent.
  2. Low-Level Ventral Nerve Cord (VNC):
     Constrained strictly by the Janelia MaleCNS v1.0 biological connectome. Receives the
     descending command z and maps it through 125 interneurons and 377 motor neurons to
     coordinate all 42 leg joints.
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


class HighLevelBrain(nn.Module):
    """Executive brain controller mapping sensory observations to descending commands.
    
    In biological Drosophila, the central brain (visual system, central complex, antenna lobes)
    processes external stimuli and issues steering and velocity instructions through a tiny
    population of Descending Neurons (DNs) that pass through the neck connective.
    """

    def __init__(
        self,
        obs_dim: int,
        num_dn: int = 4,
        hidden_dim: int = 64,
        activation: type[nn.Module] = nn.Tanh,
    ) -> None:
        """Initialize the high-level brain controller.
        
        Args:
            obs_dim: Dimension of sensory observation vector.
            num_dn: Number of descending neurons (4 for DNa01/DNa02).
            hidden_dim: Hidden dimension of brain MLP.
            activation: PyTorch activation module (e.g. nn.Tanh).
        """
        super().__init__()
        self.obs_dim = obs_dim
        self.num_dn = num_dn

        self.net = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            activation(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            activation(),
            layer_init(nn.Linear(hidden_dim, num_dn)),
            nn.Tanh(),  # Normalized descending commands in [-1, +1]
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Compute descending command vector z from sensory observation.
        
        Args:
            obs: Observation tensor of shape (batch_size, obs_dim).
            
        Returns:
            z: Descending command tensor of shape (batch_size, num_dn).
        """
        return self.net(obs)


class LowLevelVNC(nn.Module):
    """Ventral Nerve Cord (VNC) motor controller constrained by the biological connectome.
    
    Receives 4-dimensional descending command signals z from the brain and propagates them
    along biological synapses:
      4 DNs -> 125 VNC Interneurons -> 377 Motor Neurons -> 42 Joint Actuators
    """

    def __init__(
        self,
        circuit_data: ConnectomeCircuitData,
        act_dim: int = 42,
        activation: type[nn.Module] = nn.Tanh,
        use_biological_weights: bool = True,
    ) -> None:
        """Initialize the connectome-constrained VNC controller.
        
        Args:
            circuit_data: ConnectomeCircuitData holding biological masks and synapse weights.
            act_dim: Number of actuated leg joint DOFs (42 for FlyGym).
            activation: PyTorch activation module (e.g. nn.Tanh).
            use_biological_weights: If True, initialize weights with log-scaled synapse counts.
        """
        super().__init__()
        self.circuit_data = circuit_data
        self.act_dim = act_dim

        num_dn = len(circuit_data.dn_indices)
        num_in = len(circuit_data.in_indices)
        num_mn = len(circuit_data.mn_indices)

        # Hop 1 Layer: 4 DNs -> 125 VNC Interneurons (MaskedLinear)
        init_hop1_w = circuit_data.hop1_weights if use_biological_weights else None
        self.hop1_layer = MaskedLinear(
            in_features=num_dn,
            out_features=num_in,
            mask=circuit_data.hop1_mask,
            initial_weights=init_hop1_w,
        )
        self.act1 = activation()

        # Hop 2 Layer: 125 VNC Interneurons -> 377 Motor Neurons (MaskedLinear)
        init_hop2_w = circuit_data.hop2_weights if use_biological_weights else None
        self.hop2_layer = MaskedLinear(
            in_features=num_in,
            out_features=num_mn,
            mask=circuit_data.hop2_mask,
            initial_weights=init_hop2_w,
        )
        self.act2 = activation()

        # Actuator Decoder: 377 Motor Neurons -> 42 leg joint commands
        self.actuator_decoder = layer_init(nn.Linear(num_mn, act_dim), std=0.01)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Map descending commands through the connectome into leg joint actions.
        
        Args:
            z: Descending command tensor of shape (batch_size, num_dn).
            
        Returns:
            action_mean: Joint command vector of shape (batch_size, act_dim).
        """
        h_in = self.act1(self.hop1_layer(z))
        h_mn = self.act2(self.hop2_layer(h_in))
        return self.actuator_decoder(h_mn)

    def lesion_dn_channel(self, channel_idx: int) -> int:
        """Ablate a specific descending neuron channel (0 to 3) in Hop 1."""
        return self.hop1_layer.apply_lesion(target_cols=[channel_idx])

    def lesion_vnc_interneurons(self, in_indices: list[int]) -> tuple[int, int]:
        """Ablate specific VNC interneurons from both input and output sides."""
        silenced_in = self.hop1_layer.apply_lesion(target_rows=in_indices)
        silenced_out = self.hop2_layer.apply_lesion(target_cols=in_indices)
        return silenced_in, silenced_out

    def reset_lesions(self) -> None:
        """Restore all severed connections to their intact biological state."""
        self.hop1_layer.reset_lesions()
        self.hop2_layer.reset_lesions()


class HierarchicalPolicy(nn.Module):
    """Two-tier Hierarchical Actor-Critic Policy (Brain -> VNC Connectome).
    
    Combines:
      1. High-Level Cephalic Brain: Maps raw sensory observations to descending commands z.
      2. Low-Level VNC: Translates z into 42 leg joint actions via biological connectome.
      3. Critic: Value function estimating expected cumulative returns V(s).
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        circuit_data: ConnectomeCircuitData | str | Path,
        brain_hidden_dim: int = 64,
        critic_hidden_dim: int = 128,
        activation: str = "tanh",
        init_log_std: float = -0.5,
        use_biological_weights: bool = True,
    ) -> None:
        """Initialize the Hierarchical Policy.
        
        Args:
            obs_dim: Dimension of sensory observation vector.
            act_dim: Dimension of action vector (42 for FlyGym).
            circuit_data: ConnectomeCircuitData object or path to .pt file.
            brain_hidden_dim: Hidden dimension for high-level brain MLP.
            critic_hidden_dim: Hidden dimension for value network MLP.
            activation: Activation function ('tanh' or 'relu').
            init_log_std: Initial exploration standard deviation.
            use_biological_weights: If True, initialize with biological synapse counts.
        """
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        if isinstance(circuit_data, (str, Path)):
            circuit_data = load_circuit_data(circuit_data)
        self.circuit_data = circuit_data

        act_cls = nn.Tanh if activation.lower() == "tanh" else nn.ReLU
        num_dn = len(circuit_data.dn_indices)

        # 1. High-Level Cephalic Brain
        self.brain = HighLevelBrain(
            obs_dim=obs_dim,
            num_dn=num_dn,
            hidden_dim=brain_hidden_dim,
            activation=act_cls,
        )

        # 2. Low-Level Biological VNC
        self.vnc = LowLevelVNC(
            circuit_data=circuit_data,
            act_dim=act_dim,
            activation=act_cls,
            use_biological_weights=use_biological_weights,
        )

        # Learnable log-std for continuous action exploration
        self.actor_logstd = nn.Parameter(torch.ones(act_dim) * init_log_std)

        # 3. Critic (Value Network)
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, critic_hidden_dim)),
            act_cls(),
            layer_init(nn.Linear(critic_hidden_dim, critic_hidden_dim)),
            act_cls(),
            layer_init(nn.Linear(critic_hidden_dim, 1), std=1.0),
        )

    # --- Decoupled Hierarchical Interface ---

    def get_descending_command(self, obs: torch.Tensor) -> torch.Tensor:
        """Extract the 4-dimensional descending command issued by the brain.
        
        Useful for inspecting steering intent, forward speed intent, and logging
        brain state during locomotion.
        
        Args:
            obs: Observation tensor of shape (batch_size, obs_dim).
            
        Returns:
            z: Descending command tensor of shape (batch_size, 4).
        """
        return self.brain(obs)

    def act_from_descending_command(self, z: torch.Tensor) -> torch.Tensor:
        """Evaluate the low-level VNC given a direct descending command vector z.
        
        Simulates in-silico optogenetic activation / clamping: researchers can manually
        inject descending commands (e.g. z = [1.0, -1.0, 0.5, 0.5] for turning) to test
        how the biological VNC coordinates leg joints in isolation.
        
        Args:
            z: Descending command tensor of shape (batch_size, 4).
            
        Returns:
            action_mean: Deterministic joint target angles of shape (batch_size, 42).
        """
        return self.vnc(z)

    # --- Reinforcement Learning Interface ---

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Estimate the state-value V(s) using the critic network."""
        return self.critic(obs)

    def get_deterministic_action(self, obs: torch.Tensor) -> torch.Tensor:
        """Compute the deterministic mean action (for testing and evaluation)."""
        z = self.get_descending_command(obs)
        return self.act_from_descending_command(z)

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Actor-Critic forward step for PPO training.
        
        Args:
            obs: Batch of observations of shape (batch_size, obs_dim).
            action: Optional existing actions to evaluate log probability for.
            
        Returns:
            action: Sampled action tensor of shape (batch_size, act_dim).
            log_prob: Log probability of the action of shape (batch_size,).
            entropy: Policy entropy of shape (batch_size,).
            value: Estimated state-value of shape (batch_size, 1).
        """
        # High-level brain computes descending command z
        z = self.brain(obs)

        # Low-level VNC maps z into leg joint targets
        action_mean = self.vnc(z)
        action_std = torch.exp(self.actor_logstd)

        # Diagonal Gaussian distribution over the 42 continuous joint actions
        dist = Normal(action_mean, action_std)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action).sum(axis=-1)
        entropy = dist.entropy().sum(axis=-1)
        value = self.critic(obs)

        return action, log_prob, entropy, value

    # --- Freezing & Two-Stage Training Support ---

    def freeze_vnc(self) -> None:
        """Freeze low-level VNC connectome weights (useful when training only the brain)."""
        for param in self.vnc.parameters():
            param.requires_grad = False

    def unfreeze_vnc(self) -> None:
        """Unfreeze low-level VNC connectome weights."""
        for param in self.vnc.parameters():
            param.requires_grad = True

    def freeze_brain(self) -> None:
        """Freeze high-level cephalic brain weights (useful when pre-training the VNC)."""
        for param in self.brain.parameters():
            param.requires_grad = False

    def unfreeze_brain(self) -> None:
        """Unfreeze high-level cephalic brain weights."""
        for param in self.brain.parameters():
            param.requires_grad = True

    # --- In-Silico Lesion Controls ---

    def lesion_dn_channel(self, channel_idx: int) -> int:
        """Surgically silence a specific descending neuron in the low-level VNC."""
        return self.vnc.lesion_dn_channel(channel_idx)

    def reset_lesions(self) -> None:
        """Restore all severed connections in the VNC to normal biological state."""
        self.vnc.reset_lesions()
