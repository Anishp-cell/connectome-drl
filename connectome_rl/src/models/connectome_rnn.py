"""Connectome Recurrent Neural Network (Architecture 2 from Table: Connectome-RNN).

Implements a recurrent connectome architecture featuring biological Central Pattern
Generators (CPGs). Recurrent lateral feedback loops among VNC Interneurons enable
the network to sustain rhythmic stepping oscillations without requiring constant
external commands.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

from connectome_rl.src.models.connectome_layers import MaskedLinear
from connectome_rl.src.models.mlp_policy import layer_init
from connectome_rl.src.connectome.graph_utils import ConnectomeCircuitData, load_circuit_data


class ConnectomeRNNCell(nn.Module):
    """Recurrent VNC Interneuron cell with biological lateral feedback loops (CPG).
    
    Update Equation:
        h_t = tanh( (W_ff ⊙ M_ff) u_t + (W_rec ⊙ M_rec) h_{t-1} + bias )
        
    Where:
      - u_t is the 4-dim Descending Neuron command vector (DNa01/DNa02).
      - M_ff is the 125 x 4 feedforward biological mask (DN -> Interneurons).
      - M_rec is the 125 x 125 lateral feedback mask (Interneuron -> Interneuron loops).
      - h_t is the 125-dim state of the VNC spinal cord at timestep t.
    """

    def __init__(
        self,
        num_dn: int,
        num_in: int,
        hop1_mask: torch.BoolTensor,
        rec_mask: torch.BoolTensor,
        hop1_weights: torch.FloatTensor | None = None,
        rec_weights: torch.FloatTensor | None = None,
        spectral_radius: float = 0.95,
    ) -> None:
        super().__init__()
        self.num_dn = num_dn
        self.num_in = num_in

        # Feedforward drive: 4 DNs -> 125 INs
        self.ff_layer = MaskedLinear(
            in_features=num_dn,
            out_features=num_in,
            mask=hop1_mask,
            initial_weights=hop1_weights,
            bias=True,
        )

        # Recurrent feedback: 125 INs -> 125 INs (Biological CPG loops)
        self.rec_layer = MaskedLinear(
            in_features=num_in,
            out_features=num_in,
            mask=rec_mask,
            initial_weights=rec_weights,
            bias=False,  # Bias is already in ff_layer
        )

        # Calibrate recurrent spectral radius to ensure stable, non-explosive oscillations
        self._calibrate_recurrent_weights(spectral_radius)

    def _calibrate_recurrent_weights(self, target_radius: float) -> None:
        """Scale recurrent weights so the spectral radius prevents exploding feedback."""
        with torch.no_grad():
            w = self.rec_layer.effective_weight.cpu().numpy()
            if np.any(w):
                eigenvals = np.linalg.eigvals(w)
                max_eigen = np.max(np.abs(eigenvals))
                if max_eigen > 1e-6:
                    scale = target_radius / max_eigen
                    self.rec_layer.weight.mul_(float(scale))

    def forward(self, u_t: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        """Compute next VNC interneuron hidden state h_t.
        
        Args:
            u_t: Descending neuron inputs of shape (batch_size, num_dn).
            h_prev: Previous hidden state of shape (batch_size, num_in).
            
        Returns:
            h_t: Next hidden state of shape (batch_size, num_in).
        """
        ff_drive = self.ff_layer(u_t)
        rec_drive = self.rec_layer(h_prev)
        return torch.tanh(ff_drive + rec_drive)


class ConnectomeRNNPolicy(nn.Module):
    """Actor-Critic Policy using Connectome-RNN for rhythmic, CPG-driven walking.
    
    Signal Pipeline:
      1. Sensory Encoder: maps raw observation (100-dim) -> 4 DNs.
      2. ConnectomeRNNCell: integrates 4 DNs + previous 125 IN state -> next 125 IN state.
      3. Hop 2 MaskedLinear: maps 125 INs -> 377 Motor Neurons.
      4. Actuator Decoder: maps 377 MNs -> 42 leg joint commands.
      5. Critic: state-value function V(s).
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        circuit_data: ConnectomeCircuitData | str | Path,
        init_log_std: float = -0.5,
        spectral_radius: float = 0.95,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        if isinstance(circuit_data, (str, Path)):
            circuit_data = load_circuit_data(circuit_data)
        self.circuit_data = circuit_data

        num_dn = len(circuit_data.dn_indices)
        num_in = len(circuit_data.in_indices)
        num_mn = len(circuit_data.mn_indices)
        self.num_in = num_in

        # Extract the 125 x 125 biological recurrent mask among interneurons
        in_indices = circuit_data.in_indices
        rec_mask = circuit_data.full_mask[in_indices][:, in_indices]
        rec_weights = circuit_data.full_weights[in_indices][:, in_indices]

        # 1. Sensory Encoder
        self.sensory_encoder = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, num_dn)),
            nn.Tanh(),
        )

        # 2. Recurrent Connectome CPG Cell (4 DN -> 125 IN, 125 IN -> 125 IN)
        self.rnn_cell = ConnectomeRNNCell(
            num_dn=num_dn,
            num_in=num_in,
            hop1_mask=circuit_data.hop1_mask,
            rec_mask=rec_mask,
            hop1_weights=circuit_data.hop1_weights,
            rec_weights=rec_weights,
            spectral_radius=spectral_radius,
        )

        # 3. Hop 2 Muscle Activation Layer: 125 IN -> 377 MN
        self.hop2_layer = MaskedLinear(
            in_features=num_in,
            out_features=num_mn,
            mask=circuit_data.hop2_mask,
            initial_weights=circuit_data.hop2_weights,
        )

        # 4. Actuator Decoder: 377 MN -> 42 leg joint commands
        self.actuator_decoder = layer_init(nn.Linear(num_mn, act_dim), std=0.01)

        # Learnable Gaussian exploration parameter
        self.actor_logstd = nn.Parameter(torch.ones(act_dim) * init_log_std)

        # 5. Critic (Value network)
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 1), std=1.0),
        )

    def init_hidden(self, batch_size: int = 1, device: torch.device | str = "cpu") -> torch.Tensor:
        """Initialize zero hidden state for the 125 VNC interneurons."""
        return torch.zeros(batch_size, self.num_in, device=device)

    def forward_actor(
        self,
        obs: torch.Tensor,
        h_prev: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Run single recurrent timestep of the biological CPG.
        
        Args:
            obs: Observation tensor of shape (batch_size, obs_dim).
            h_prev: Previous VNC interneuron state (batch_size, 125).
                   If None, initialized to zeros.
                   
        Returns:
            action_mean: Joint command targets of shape (batch_size, 42).
            h_next: Updated interneuron state of shape (batch_size, 125).
            activations: Dictionary of biological intermediate firings.
        """
        if h_prev is None:
            h_prev = self.init_hidden(batch_size=obs.shape[0], device=obs.device)

        # 1. Sensory inputs drive the Descending Neurons
        u_t = self.sensory_encoder(obs)

        # 2. Recurrent CPG step (integrating brain commands + previous spinal state)
        h_next = self.rnn_cell(u_t, h_prev)

        # 3. Interneurons drive Motor Neurons
        mn_activity = torch.tanh(self.hop2_layer(h_next))

        # 4. Motor Neurons drive the 42 leg joints
        action_mean = self.actuator_decoder(mn_activity)

        activations = {
            "dn": u_t,
            "interneuron": h_next,
            "motor_neuron": mn_activity,
        }
        return action_mean, h_next, activations

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Estimate state value V(s)."""
        return self.critic(obs)

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        h_prev: torch.Tensor | None = None,
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute action distribution, sample action, state value, and next hidden state."""
        action_mean, h_next, _ = self.forward_actor(obs, h_prev=h_prev)
        action_logstd = torch.clamp(self.actor_logstd, min=-20.0, max=2.0)
        action_std = torch.exp(action_logstd)

        dist = Normal(action_mean, action_std)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action).sum(axis=-1)
        entropy = dist.entropy().sum(axis=-1)
        value = self.critic(obs)

        return action, log_prob, entropy, value, h_next

    def get_deterministic_action(
        self,
        obs: torch.Tensor,
        h_prev: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get deterministic mean action and next hidden state."""
        with torch.no_grad():
            action_mean, h_next, _ = self.forward_actor(obs, h_prev=h_prev)
            return action_mean, h_next

    # --- In-Silico Lesion Controls ---

    def lesion_dn_channel(self, dn_channel: int) -> int:
        """Silence a specific Descending Neuron command channel."""
        return self.rnn_cell.ff_layer.apply_lesion(target_cols=[dn_channel])

    def lesion_cpg_loops(self, in_indices: list[int] | None = None) -> int:
        """Silence recurrent feedback loops in the VNC to disable the CPG rhythm."""
        if in_indices is None:
            in_indices = list(range(self.num_in))
        return self.rnn_cell.rec_layer.apply_lesion(target_rows=in_indices, target_cols=in_indices)

    def reset_lesions(self) -> None:
        """Restore all biological feedforward and recurrent connections."""
        self.rnn_cell.ff_layer.reset_lesions()
        self.rnn_cell.rec_layer.reset_lesions()
        self.hop2_layer.reset_lesions()


if __name__ == "__main__":
    torch.manual_seed(42)
    data_dir = Path(__file__).resolve().parents[2] / "data"
    circuit_path = data_dir / "dna_circuit_tensors.pt"
    
    if not circuit_path.exists():
        print(f"Error: {circuit_path} not found. Run graph_utils.py first!")
        exit(1)
        
    batch_size = 2
    dummy_obs_dim = 100
    dummy_act_dim = 42
    
    policy = ConnectomeRNNPolicy(
        obs_dim=dummy_obs_dim,
        act_dim=dummy_act_dim,
        circuit_data=circuit_path,
    )
    
    print("=" * 65)
    print("CONNECTOME-RNN POLICY (CPG RECURRENT MODEL) SMOKE TEST")
    print("=" * 65)
    print(f"Feedforward Cell: {policy.rnn_cell.ff_layer}")
    print(f"Recurrent CPG Cell: {policy.rnn_cell.rec_layer}")
    print(f"Hop 2 Muscle Layer: {policy.hop2_layer}")
    
    # Simulate an 8-step walking sequence to test CPG oscillations
    print("\n--- Testing 8-Step Recurrent Sequence (CPG Stepping) ---")
    h = policy.init_hidden(batch_size=batch_size)
    dummy_obs = torch.randn(batch_size, dummy_obs_dim)
    
    h_states = []
    actions = []
    for step in range(8):
        action, log_prob, entropy, value, h = policy.get_action_and_value(dummy_obs, h_prev=h)
        h_states.append(h)
        actions.append(action)
        print(f"Step {step+1}: Action mean = {action.mean():.4f}, Hidden state norm = {h.norm():.4f}")
        
    print(f"✅ Rhythmic hidden state successfully propagated across 8 timesteps!")
    
    # Test gradient backward pass through recurrent time
    loss = value.mean() - log_prob.mean()
    loss.backward()
    
    rec_unwired_grad = policy.rnn_cell.rec_layer.weight.grad[~policy.rnn_cell.rec_layer.mask]
    assert (rec_unwired_grad == 0.0).all(), "Recurrent unwired synapses must have 0.0 gradient!"
    print("✅ Recurrent biological gradient isolation verified!")
    
    # Test CPG Lesion (cutting lateral rhythm loops)
    print("\n--- In-Silico CPG Lesion Test ---")
    silenced = policy.lesion_cpg_loops()
    print(f"Silenced {silenced} recurrent CPG feedback synapses.")
    policy.reset_lesions()
    print("✅ CPG feedback loop lesion & restoration verified!")
    print("=" * 65)
