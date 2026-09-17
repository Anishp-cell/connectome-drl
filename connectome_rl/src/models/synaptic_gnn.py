"""Synaptic Graph Neural Network (Architecture 3 from Table: SynapticGNN).

Implements a pure PyTorch Message Passing Neural Network directly on the
MaleCNS connectome topology. Each of the 506 neurons is an individual graph node,
and information propagates along the 1,581 biological synapses via directed
message passing with synaptic weight scaling.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

from connectome_rl.src.models.mlp_policy import layer_init
from connectome_rl.src.connectome.graph_utils import ConnectomeCircuitData, load_circuit_data


class SynapticMessagePassingLayer(nn.Module):
    """Message passing layer along biological connectome edges.
    
    Equations:
      1. Message:    m_{u->v} = (W_msg · h_u) * edge_weight_{uv} * edge_lesion_mask_{uv}
      2. Aggregate:  m_agg_v  = Σ_{u ∈ N_in(v)} m_{u->v}
      3. Update:     h_v'     = tanh( W_self · h_v + m_agg_v + bias )
    """

    def __init__(
        self,
        node_dim: int,
        edge_index: torch.LongTensor,
        edge_weight: torch.FloatTensor,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.node_dim = node_dim

        # Register edge topology as buffers (source -> target)
        self.register_buffer("edge_index", edge_index.long())
        self.register_buffer("edge_weight", edge_weight.float())
        
        # Dynamic lesion mask for edges (1.0 = intact, 0.0 = severed)
        self.register_buffer("edge_lesion_mask", torch.ones_like(edge_weight, dtype=torch.float32))

        # Learnable message projection and self-update projection
        self.msg_linear = nn.Linear(node_dim, node_dim, bias=False)
        self.self_linear = nn.Linear(node_dim, node_dim, bias=bias)

        # Initialize weights
        nn.init.orthogonal_(self.msg_linear.weight, gain=1.0)
        nn.init.orthogonal_(self.self_linear.weight, gain=1.0)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Propagate messages along connectome synapses.
        
        Args:
            h: Node embeddings of shape (batch_size, num_nodes, node_dim).
            
        Returns:
            h_next: Updated node embeddings of shape (batch_size, num_nodes, node_dim).
        """
        batch_size, num_nodes, node_dim = h.shape
        src_nodes = self.edge_index[0]  # Shape: (num_edges,)
        tgt_nodes = self.edge_index[1]  # Shape: (num_edges,)

        # Step 1: Compute messages from source nodes
        # Shape: (batch_size, num_nodes, node_dim)
        h_projected = self.msg_linear(h)
        
        # Gather source representations for every edge: (batch_size, num_edges, node_dim)
        src_messages = h_projected[:, src_nodes, :]

        # Scale messages by biological synapse strength and lesion mask
        # Shape: (1, num_edges, 1)
        effective_weights = (self.edge_weight * self.edge_lesion_mask).unsqueeze(0).unsqueeze(-1)
        scaled_messages = src_messages * effective_weights

        # Step 2: Aggregate incoming messages at target nodes using index_add
        aggregated = torch.zeros(batch_size, num_nodes, node_dim, device=h.device, dtype=h.dtype)
        
        # Transpose to (num_nodes, batch_size, node_dim) for scatter aggregation
        agg_t = aggregated.transpose(0, 1)
        msg_t = scaled_messages.transpose(0, 1)
        agg_t.index_add_(0, tgt_nodes, msg_t)
        aggregated = agg_t.transpose(0, 1)

        # Step 3: Combine self-state and incoming aggregated messages
        self_state = self.self_linear(h)
        h_next = torch.tanh(self_state + aggregated)
        return h_next


class SynapticGNNPolicy(nn.Module):
    """Actor-Critic Policy using Synaptic Graph Neural Network message passing.
    
    Signal Pipeline:
      1. Sensory Encoder: maps raw sensory inputs (100-dim) to the 4 Descending Neurons.
      2. Graph Embedding: initializes all 506 neurons with feature vectors (node_dim).
      3. Message Passing: runs K iterations (hops) of synaptic message passing along the
         1,581 biological connections.
      4. Actuator Readout: pools the 377 Motor Neuron embeddings and projects them
         to the 42 continuous leg joint commands.
      5. Critic: state-value function V(s).
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        circuit_data: ConnectomeCircuitData | str | Path,
        node_dim: int = 16,
        num_hops: int = 2,
        init_log_std: float = -0.5,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.node_dim = node_dim
        self.num_hops = num_hops

        if isinstance(circuit_data, (str, Path)):
            circuit_data = load_circuit_data(circuit_data)
        self.circuit_data = circuit_data

        self.num_nodes = circuit_data.num_nodes
        self.dn_indices = circuit_data.dn_indices
        self.mn_indices = circuit_data.mn_indices

        # 1. Sensory Encoder: maps sensory inputs (100) -> 4 DN command features
        self.sensory_encoder = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, len(self.dn_indices) * node_dim)),
            nn.Tanh(),
        )

        # Learnable baseline resting potential for all 506 neurons
        self.base_node_embeddings = nn.Parameter(torch.randn(self.num_nodes, node_dim) * 0.05)

        # 2. Synaptic Message Passing Layers (K hops)
        self.gnn_layers = nn.ModuleList([
            SynapticMessagePassingLayer(
                node_dim=node_dim,
                edge_index=circuit_data.edge_index,
                edge_weight=circuit_data.edge_weight,
            )
            for _ in range(num_hops)
        ])

        # 3. Actuator Readout: maps 377 Motor Neurons (pooled or flattened) -> 42 joint targets
        num_mn = len(self.mn_indices)
        self.actuator_readout = nn.Sequential(
            layer_init(nn.Linear(num_mn * node_dim, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, act_dim), std=0.01),
        )

        # Learnable Gaussian exploration parameter
        self.actor_logstd = nn.Parameter(torch.ones(act_dim) * init_log_std)

        # 4. Critic (Value network)
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 1), std=1.0),
        )

    def forward_actor(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Propagate signals through the biological graph.
        
        Returns:
            action_mean: Joint command targets of shape (batch_size, 42).
            final_node_embeddings: Node features of shape (batch_size, 506, node_dim).
        """
        batch_size = obs.shape[0]

        # Step 1: Initialize all 506 nodes with baseline resting embeddings
        # Shape: (batch_size, 506, node_dim)
        h = self.base_node_embeddings.unsqueeze(0).expand(batch_size, -1, -1).clone()

        # Step 2: Inject sensory commands into the 4 Descending Neurons
        dn_features = self.sensory_encoder(obs).view(batch_size, len(self.dn_indices), self.node_dim)
        h[:, self.dn_indices, :] = h[:, self.dn_indices, :] + dn_features

        # Step 3: Run K hops of synaptic message passing
        for gnn_layer in self.gnn_layers:
            h = gnn_layer(h)

        # Step 4: Extract the 377 Motor Neurons and decode to 42 leg joints
        mn_features = h[:, self.mn_indices, :]  # Shape: (batch_size, 377, node_dim)
        mn_flat = mn_features.reshape(batch_size, -1)
        action_mean = self.actuator_readout(mn_flat)

        return action_mean, h

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Estimate state value V(s)."""
        return self.critic(obs)

    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute action distribution, sample action, and estimate state value."""
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

    def lesion_nodes(self, node_indices: list[int]) -> int:
        """Silence all synaptic messages originating from or targeting specific neurons."""
        target_set = set(node_indices)
        silenced_total = 0
        
        for gnn_layer in self.gnn_layers:
            edge_index = gnn_layer.edge_index
            # Find edges where source or target is in target_set
            src_in = torch.tensor([s.item() in target_set for s in edge_index[0]], device=edge_index.device)
            tgt_in = torch.tensor([t.item() in target_set for t in edge_index[1]], device=edge_index.device)
            lesion_mask = ~(src_in | tgt_in)
            
            silenced_edges = (~lesion_mask).sum().item()
            gnn_layer.edge_lesion_mask.copy_(lesion_mask.float())
            silenced_total += silenced_edges
            
        return silenced_total

    def reset_lesions(self) -> None:
        """Restore all severed synaptic connections."""
        for gnn_layer in self.gnn_layers:
            gnn_layer.edge_lesion_mask.fill_(1.0)


if __name__ == "__main__":
    torch.manual_seed(42)
    data_dir = Path(__file__).resolve().parents[2] / "data"
    circuit_path = data_dir / "dna_circuit_tensors.pt"
    
    if not circuit_path.exists():
        print(f"Error: {circuit_path} not found. Run graph_utils.py first!")
        exit(1)
        
    batch_size = 4
    dummy_obs_dim = 100
    dummy_act_dim = 42
    
    policy = SynapticGNNPolicy(
        obs_dim=dummy_obs_dim,
        act_dim=dummy_act_dim,
        circuit_data=circuit_path,
        node_dim=16,
        num_hops=2,
    )
    
    print("=" * 65)
    print("SYNAPTIC GNN POLICY (GRAPH NEURAL NETWORK MODEL) SMOKE TEST")
    print("=" * 65)
    print(f"Total Graph Nodes:       {policy.num_nodes}")
    print(f"Message Passing Hops:    {policy.num_hops}")
    print(f"Synaptic Edges per Hop:  {policy.gnn_layers[0].edge_index.shape[1]}")
    
    dummy_obs = torch.randn(batch_size, dummy_obs_dim)
    action, log_prob, entropy, value = policy.get_action_and_value(dummy_obs)
    
    print(f"\nForward output action shape: {action.shape}")
    print(f"State value V(s) shape:      {value.shape}")
    
    # Backward pass gradient test
    loss = value.mean() - log_prob.mean()
    loss.backward()
    
    has_grads = any(p.grad is not None and not torch.isnan(p.grad).any() for p in policy.parameters())
    print(f"✅ GNN gradient backpropagation verified: {has_grads}")
    
    # In-silico lesion test
    print("\n--- In-Silico Node Lesion Test (Silencing DNa01) ---")
    intact_action = policy.get_deterministic_action(dummy_obs)
    silenced_edges = policy.lesion_nodes([policy.dn_indices[0]])
    print(f"Severed {silenced_edges} synaptic connections to/from DNa01 across {policy.num_hops} hops.")
    
    lesioned_action = policy.get_deterministic_action(dummy_obs)
    diff = (lesioned_action - intact_action).abs().mean().item()
    print(f"Mean motor output difference under GNN lesion: {diff:.4f}")
    assert diff > 0.0, "GNN lesion must alter leg motor outputs!"
    
    policy.reset_lesions()
    restored_action = policy.get_deterministic_action(dummy_obs)
    assert torch.allclose(restored_action, intact_action), "Resetting lesions must restore normal function!"
    print("✅ GNN lesion and recovery verified!")
    print("=" * 65)
