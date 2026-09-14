"""Graph Utilities for Embodied Connectomics.

Converts NetworkX biological connectome graphs into PyTorch tensors:
1. Node ID to 0-indexed tensor mappings.
2. Binary boolean adjacency masks (torch.BoolTensor) for MaskedLinear layers.
3. Hierarchical bipartite masks (DN -> Interneurons, Interneurons -> Motor Neurons).
4. PyTorch Geometric edge indices (edge_index) and edge attributes for GNNs.
5. Sparse COO tensor representations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class ConnectomeCircuitData:
    """Container holding PyTorch-ready representations of a connectome subcircuit."""
    
    # Node count and ID mappings
    num_nodes: int
    node_to_idx: dict[int, int]
    idx_to_node: dict[int, int]
    
    # Role-specific index lists
    dn_indices: list[int]
    in_indices: list[int]
    mn_indices: list[int]
    
    # Full (N x N) connectivity
    full_mask: torch.BoolTensor          # Shape: (num_nodes, num_nodes)
    full_weights: torch.FloatTensor      # Shape: (num_nodes, num_nodes)
    
    # Hierarchical 2-hop bipartite masks for layered feedforward policies
    # Hop 1: DN -> Interneurons (Shape: num_in, num_dn)
    hop1_mask: torch.BoolTensor
    hop1_weights: torch.FloatTensor
    
    # Hop 2: Interneurons -> Motor Neurons (Shape: num_mn, num_in)
    hop2_mask: torch.BoolTensor
    hop2_weights: torch.FloatTensor
    
    # PyG / GNN representations (source -> target format)
    edge_index: torch.LongTensor         # Shape: (2, num_edges)
    edge_weight: torch.FloatTensor       # Shape: (num_edges,)


def build_node_mappings(graph: nx.DiGraph) -> tuple[dict[int, int], dict[int, int], dict[str, list[int]]]:
    """Map arbitrary biological bodyIds to contiguous 0-indexed tensor row/cols.
    
    Orders nodes deterministically: Descending Neurons first, then Interneurons,
    then Motor Neurons.
    
    Args:
        graph: NetworkX DiGraph with 'role' node attribute.
        
    Returns:
        tuple of (node_to_idx, idx_to_node, role_indices_dict)
    """
    dns = sorted([n for n, d in graph.nodes(data=True) if d.get("role") == "DN"])
    ins = sorted([n for n, d in graph.nodes(data=True) if d.get("role") not in ("DN", "MN")])
    mns = sorted([n for n, d in graph.nodes(data=True) if d.get("role") == "MN"])
    
    ordered_nodes = dns + ins + mns
    node_to_idx = {node: idx for idx, node in enumerate(ordered_nodes)}
    idx_to_node = {idx: node for node, idx in node_to_idx.items()}
    
    role_indices = {
        "DN": [node_to_idx[n] for n in dns],
        "VNC_IN": [node_to_idx[n] for n in ins],
        "MN": [node_to_idx[n] for n in mns],
    }
    
    logger.info(
        f"Indexed {len(ordered_nodes)} neurons: "
        f"{len(dns)} DNs, {len(ins)} Interneurons, {len(mns)} MNs"
    )
    return node_to_idx, idx_to_node, role_indices


def graph_to_full_adjacency(
    graph: nx.DiGraph,
    node_to_idx: dict[int, int],
    weight_attr: str = "norm_weight",
) -> tuple[torch.BoolTensor, torch.FloatTensor]:
    """Build full (N, N) boolean mask and weight tensor.
    
    Convention: target_row, source_col (Y = W @ X).
    mask[i, j] = True means neuron j sends a synapse to neuron i.
    
    Args:
        graph: NetworkX DiGraph.
        node_to_idx: Node mapping dictionary.
        weight_attr: Edge attribute to use for synaptic weight values.
        
    Returns:
        Tuple of (bool_mask, float_weights) of shape (N, N).
    """
    n = len(node_to_idx)
    mask = torch.zeros((n, n), dtype=torch.bool)
    weights = torch.zeros((n, n), dtype=torch.float32)
    
    for u, v, data in graph.edges(data=True):
        src_idx = node_to_idx[u]
        tgt_idx = node_to_idx[v]
        w = float(data.get(weight_attr, data.get("weight", 1.0)))
        
        mask[tgt_idx, src_idx] = True
        weights[tgt_idx, src_idx] = w
        
    return mask, weights


def graph_to_hierarchical_masks(
    graph: nx.DiGraph,
    node_to_idx: dict[int, int],
    role_indices: dict[str, list[int]],
    weight_attr: str = "norm_weight",
) -> tuple[torch.BoolTensor, torch.FloatTensor, torch.BoolTensor, torch.FloatTensor]:
    """Construct bipartite matrices for 2-tier motor policies:
    
    Layer 1 (DN -> Interneurons):
        Shape: (num_interneurons, num_dns)
    Layer 2 (Interneurons -> Motor Neurons):
        Shape: (num_mns, num_interneurons)
        
    Args:
        graph: NetworkX DiGraph.
        node_to_idx: Global node-to-index mapping.
        role_indices: Role lists containing global indices.
        weight_attr: Edge weight attribute name.
        
    Returns:
        (hop1_mask, hop1_weights, hop2_mask, hop2_weights)
    """
    dn_globals = role_indices["DN"]
    in_globals = role_indices["VNC_IN"]
    mn_globals = role_indices["MN"]
    
    # Local index lookups within each layer
    dn_local = {g_idx: l_idx for l_idx, g_idx in enumerate(dn_globals)}
    in_local = {g_idx: l_idx for l_idx, g_idx in enumerate(in_globals)}
    mn_local = {g_idx: l_idx for l_idx, g_idx in enumerate(mn_globals)}
    
    num_dn = len(dn_globals)
    num_in = len(in_globals)
    num_mn = len(mn_globals)
    
    hop1_mask = torch.zeros((num_in, num_dn), dtype=torch.bool)
    hop1_weights = torch.zeros((num_in, num_dn), dtype=torch.float32)
    
    hop2_mask = torch.zeros((num_mn, num_in), dtype=torch.bool)
    hop2_weights = torch.zeros((num_mn, num_in), dtype=torch.float32)
    
    for u, v, data in graph.edges(data=True):
        src_g = node_to_idx[u]
        tgt_g = node_to_idx[v]
        w = float(data.get(weight_attr, data.get("weight", 1.0)))
        
        # Hop 1: DN -> IN
        if src_g in dn_local and tgt_g in in_local:
            r = in_local[tgt_g]
            c = dn_local[src_g]
            hop1_mask[r, c] = True
            hop1_weights[r, c] = w
            
        # Hop 2: IN -> MN
        elif src_g in in_local and tgt_g in mn_local:
            r = mn_local[tgt_g]
            c = in_local[src_g]
            hop2_mask[r, c] = True
            hop2_weights[r, c] = w

    logger.info(
        f"Hierarchical masks built: Hop 1 (DN->IN) density = {hop1_mask.float().mean():.3f}, "
        f"Hop 2 (IN->MN) density = {hop2_mask.float().mean():.3f}"
    )
    return hop1_mask, hop1_weights, hop2_mask, hop2_weights


def graph_to_edge_index(
    graph: nx.DiGraph,
    node_to_idx: dict[int, int],
    weight_attr: str = "norm_weight",
) -> tuple[torch.LongTensor, torch.FloatTensor]:
    """Convert NetworkX graph into PyTorch Geometric format (edge_index, edge_weight).
    
    Args:
        graph: NetworkX DiGraph.
        node_to_idx: Mapping from node IDs to tensor indices.
        weight_attr: Edge weight attribute name.
        
    Returns:
        edge_index: Tensor of shape (2, num_edges)
        edge_weight: Tensor of shape (num_edges,)
    """
    sources = []
    targets = []
    weights = []
    
    for u, v, data in graph.edges(data=True):
        sources.append(node_to_idx[u])
        targets.append(node_to_idx[v])
        weights.append(float(data.get(weight_attr, data.get("weight", 1.0))))
        
    edge_index = torch.tensor([sources, targets], dtype=torch.long)
    edge_weight = torch.tensor(weights, dtype=torch.float32)
    return edge_index, edge_weight


def convert_graph_to_circuit_data(
    graph: nx.DiGraph,
    weight_attr: str = "norm_weight",
) -> ConnectomeCircuitData:
    """Master factory function: Converts a NetworkX graph into ConnectomeCircuitData.
    
    Args:
        graph: Cleaned and normalized NetworkX DiGraph.
        weight_attr: Edge weight attribute to use.
        
    Returns:
        Complete ConnectomeCircuitData bundle.
    """
    node_to_idx, idx_to_node, role_indices = build_node_mappings(graph)
    full_mask, full_weights = graph_to_full_adjacency(graph, node_to_idx, weight_attr=weight_attr)
    
    hop1_mask, hop1_weights, hop2_mask, hop2_weights = graph_to_hierarchical_masks(
        graph, node_to_idx, role_indices, weight_attr=weight_attr
    )
    
    edge_index, edge_weight = graph_to_edge_index(graph, node_to_idx, weight_attr=weight_attr)
    
    return ConnectomeCircuitData(
        num_nodes=len(node_to_idx),
        node_to_idx=node_to_idx,
        idx_to_node=idx_to_node,
        dn_indices=role_indices["DN"],
        in_indices=role_indices["VNC_IN"],
        mn_indices=role_indices["MN"],
        full_mask=full_mask,
        full_weights=full_weights,
        hop1_mask=hop1_mask,
        hop1_weights=hop1_weights,
        hop2_mask=hop2_mask,
        hop2_weights=hop2_weights,
        edge_index=edge_index,
        edge_weight=edge_weight,
    )


def save_circuit_data(circuit_data: ConnectomeCircuitData, save_path: str | Path) -> None:
    """Serialize the ConnectomeCircuitData container to disk as a portable dictionary."""
    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "num_nodes": circuit_data.num_nodes,
        "node_to_idx": circuit_data.node_to_idx,
        "idx_to_node": circuit_data.idx_to_node,
        "dn_indices": circuit_data.dn_indices,
        "in_indices": circuit_data.in_indices,
        "mn_indices": circuit_data.mn_indices,
        "full_mask": circuit_data.full_mask,
        "full_weights": circuit_data.full_weights,
        "hop1_mask": circuit_data.hop1_mask,
        "hop1_weights": circuit_data.hop1_weights,
        "hop2_mask": circuit_data.hop2_mask,
        "hop2_weights": circuit_data.hop2_weights,
        "edge_index": circuit_data.edge_index,
        "edge_weight": circuit_data.edge_weight,
    }
    torch.save(payload, p)
    logger.info(f"Saved ConnectomeCircuitData tensors to {p}")


def load_circuit_data(load_path: str | Path) -> ConnectomeCircuitData:
    """Load ConnectomeCircuitData from a .pt file."""
    p = Path(load_path)
    if not p.exists():
        raise FileNotFoundError(f"Circuit tensor file not found at {p}")
    data = torch.load(p, weights_only=False)
    if isinstance(data, dict):
        return ConnectomeCircuitData(**data)
    return data


if __name__ == "__main__":
    import pickle
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    data_dir = Path(__file__).resolve().parents[2] / "data"
    pruned_graph_path = data_dir / "dna_motor_circuit_pruned.pickle"
    
    if not pruned_graph_path.exists():
        print(f"Error: {pruned_graph_path} not found. Run prune.py first!")
        exit(1)
        
    with open(pruned_graph_path, "rb") as f:
        graph = pickle.load(f)
        
    print(f"Loaded graph with {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges.")
    
    # Convert to PyTorch circuit data
    circuit_data = convert_graph_to_circuit_data(graph)
    
    # Save PyTorch bundle
    pt_save_path = data_dir / "dna_circuit_tensors.pt"
    save_circuit_data(circuit_data, pt_save_path)
    
    print("\n" + "=" * 55)
    print("PYTORCH CONNECTOME CIRCUIT TENSOR SUMMARY")
    print("=" * 55)
    print(f"Total Nodes:               {circuit_data.num_nodes}")
    print(f"DN Count (Inputs):         {len(circuit_data.dn_indices)}")
    print(f"VNC Interneurons (Hidden): {len(circuit_data.in_indices)}")
    print(f"Motor Neurons (Outputs):   {len(circuit_data.mn_indices)}")
    print(f"\nHop 1 Mask (DN -> IN):     {circuit_data.hop1_mask.shape} (Active: {circuit_data.hop1_mask.sum().item()})")
    print(f"Hop 2 Mask (IN -> MN):     {circuit_data.hop2_mask.shape} (Active: {circuit_data.hop2_mask.sum().item()})")
    print(f"Full Adjacency Tensor:     {circuit_data.full_mask.shape} (Active: {circuit_data.full_mask.sum().item()})")
    print(f"PyG Edge Index Shape:      {circuit_data.edge_index.shape}")
    print(f"\nSaved PyTorch Bundle:      {pt_save_path}")
    print("=" * 55)
