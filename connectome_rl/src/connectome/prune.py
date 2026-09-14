"""Connectome Graph Pruning and Synaptic Normalization.

Cleans biological subgraphs by:
1. Removing disconnected or dead-end neurons.
2. Thresholding weak/noisy synapses.
3. Normalizing raw synapse counts into differentiable neural network weight scales.
4. Enforcing memory constraints (<1,000 nodes).
"""

from __future__ import annotations

import copy
import logging
from typing import Literal

import networkx as nx
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def remove_orphan_nodes(graph: nx.DiGraph) -> nx.DiGraph:
    """Remove nodes with neither incoming nor outgoing connections.
    
    Args:
        graph: NetworkX DiGraph.
        
    Returns:
        Pruned DiGraph without isolated nodes.
    """
    g = graph.copy()
    isolated = [node for node in g.nodes() if g.in_degree(node) == 0 and g.out_degree(node) == 0]
    g.remove_nodes_from(isolated)
    logger.info(f"Removed {len(isolated)} orphan nodes (remaining: {g.number_of_nodes()})")
    return g


def filter_by_synapse_threshold(graph: nx.DiGraph, min_synapses: float = 3.0) -> nx.DiGraph:
    """Remove synaptic connections below the biological noise threshold.
    
    Args:
        graph: NetworkX DiGraph with 'weight' edge attribute.
        min_synapses: Minimum synapse count to retain an edge.
        
    Returns:
        DiGraph containing only edges >= min_synapses.
    """
    g = graph.copy()
    weak_edges = [
        (u, v) for u, v, d in g.edges(data=True) if d.get("weight", 0.0) < min_synapses
    ]
    g.remove_edges_from(weak_edges)
    logger.info(f"Removed {len(weak_edges)} weak edges below threshold {min_synapses}")
    # Clean any nodes that became isolated after removing weak edges
    return remove_orphan_nodes(g)


def extract_functional_pathways(
    graph: nx.DiGraph,
    source_role: str = "DN",
    target_role: str = "MN",
) -> nx.DiGraph:
    """Retain only neurons on a functional forward path from sources (DN) to targets (MN).
    
    A neuron is functional in motor control only if:
      1. It can receive signals from a Descending Neuron (DN).
      2. It can propagate signals to reach a Motor Neuron (MN).
      
    Args:
        graph: NetworkX DiGraph with 'role' node attribute.
        source_role: Attribute value for command sources (default: 'DN').
        target_role: Attribute value for actuator targets (default: 'MN').
        
    Returns:
        Pruned DiGraph containing only connected pathway nodes.
    """
    g = graph.copy()
    sources = [n for n, d in g.nodes(data=True) if d.get("role") == source_role]
    targets = [n for n, d in g.nodes(data=True) if d.get("role") == target_role]

    if not sources or not targets:
        logger.warning("Graph lacks source (DN) or target (MN) nodes. Returning input graph.")
        return g

    # Nodes reachable from at least one source (forward reachability)
    reachable_from_source = set()
    for src in sources:
        reachable_from_source.update(nx.descendants(g, src))
    reachable_from_source.update(sources)

    # Nodes that can reach at least one target (backward reachability)
    # Using reverse graph for efficient ancestor lookup
    reversed_g = g.reverse()
    can_reach_target = set()
    for tgt in targets:
        can_reach_target.update(nx.descendants(reversed_g, tgt))
    can_reach_target.update(targets)

    # Intersection: nodes that are downstream of DN AND upstream of MN
    functional_nodes = reachable_from_source.intersection(can_reach_target)

    # Create subgraph
    pruned_g = g.subgraph(functional_nodes).copy()
    removed_count = g.number_of_nodes() - pruned_g.number_of_nodes()
    logger.info(
        f"Extracted functional pathway: retained {pruned_g.number_of_nodes()} nodes, "
        f"removed {removed_count} dead-end nodes"
    )
    return pruned_g


def normalize_synapse_weights(
    graph: nx.DiGraph,
    method: Literal["log1p", "minmax", "linear_scale"] = "log1p",
    scale_factor: float = 1.0,
) -> nx.DiGraph:
    """Transform raw integer synapse counts into neural network weight scales.
    
    Raw synapse counts in MaleCNS range from 3 to 250+. In deep neural networks,
    weights this large cause immediate exploding activations and NaN gradients.
    
    Methods:
      - 'log1p': w_norm = log(1 + w) * scale_factor (compresses high counts, biological standard)
      - 'minmax': w_norm = (w - w_min) / (w_max - w_min + 1e-8)
      - 'linear_scale': w_norm = w / max_w * scale_factor
      
    Args:
        graph: NetworkX DiGraph.
        method: Normalization strategy.
        scale_factor: Multiplier to adjust overall weight initialization scale.
        
    Returns:
        DiGraph with 'norm_weight' attribute on every edge.
    """
    g = graph.copy()
    if g.number_of_edges() == 0:
        return g

    raw_weights = np.array([float(d.get("weight", 1.0)) for _, _, d in g.edges(data=True)])

    if method == "log1p":
        norm_values = np.log1p(raw_weights) * scale_factor
    elif method == "minmax":
        w_min, w_max = raw_weights.min(), raw_weights.max()
        norm_values = (raw_weights - w_min) / (w_max - w_min + 1e-8) * scale_factor + 0.05
    elif method == "linear_scale":
        norm_values = (raw_weights / (raw_weights.max() + 1e-8)) * scale_factor
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    for (u, v), nw in zip(g.edges(), norm_values):
        g[u][v]["norm_weight"] = float(nw)

    logger.info(
        f"Normalized {len(raw_weights)} edges using '{method}' "
        f"(range: [{norm_values.min():.3f}, {norm_values.max():.3f}])"
    )
    return g


def enforce_node_budget(graph: nx.DiGraph, max_nodes: int = 1000) -> nx.DiGraph:
    """Cap graph size to satisfy hardware memory constraints (<1,000 nodes).
    
    If the graph exceeds max_nodes, preserves all DNs and MNs, and ranks
    interneurons by PageRank centrality to retain only the most influential nodes.
    
    Args:
        graph: NetworkX DiGraph.
        max_nodes: Maximum allowed nodes in the graph.
        
    Returns:
        DiGraph guaranteed to have <= max_nodes.
    """
    g = graph.copy()
    if g.number_of_nodes() <= max_nodes:
        return g

    dns = [n for n, d in g.nodes(data=True) if d.get("role") == "DN"]
    mns = [n for n, d in g.nodes(data=True) if d.get("role") == "MN"]
    interneurons = [n for n, d in g.nodes(data=True) if d.get("role") not in ("DN", "MN")]

    slots_remaining = max_nodes - len(dns) - len(mns)
    if slots_remaining <= 0:
        # If DNs + MNs alone exceed budget, sample MNs by in-degree
        mn_degrees = {n: g.in_degree(n) for n in mns}
        sorted_mns = sorted(mns, key=lambda n: mn_degrees[n], reverse=True)
        retained_mns = sorted_mns[: max_nodes - len(dns)]
        keep_nodes = set(dns + retained_mns)
    else:
        # Rank interneurons by centrality
        pagerank = nx.pagerank(g, weight="weight")
        sorted_ins = sorted(interneurons, key=lambda n: pagerank.get(n, 0.0), reverse=True)
        retained_ins = sorted_ins[:slots_remaining]
        keep_nodes = set(dns + mns + retained_ins)

    pruned = g.subgraph(keep_nodes).copy()
    logger.info(f"Enforced node budget: reduced from {g.number_of_nodes()} to {pruned.number_of_nodes()}")
    return remove_orphan_nodes(pruned)


def get_graph_summary(graph: nx.DiGraph) -> dict[str, Any]:
    """Generate structural and statistical metrics for a connectome graph."""
    raw_weights = [d.get("weight", 0.0) for _, _, d in graph.edges(data=True)]
    norm_weights = [d.get("norm_weight", 0.0) for _, _, d in graph.edges(data=True)]

    roles = pd.Series([d.get("role", "Unknown") for _, d in graph.nodes(data=True)]).value_counts().to_dict()

    summary = {
        "num_nodes": graph.number_of_nodes(),
        "num_edges": graph.number_of_edges(),
        "density": nx.density(graph),
        "roles": roles,
        "raw_weight_min": float(np.min(raw_weights)) if raw_weights else 0.0,
        "raw_weight_max": float(np.max(raw_weights)) if raw_weights else 0.0,
        "raw_weight_mean": float(np.mean(raw_weights)) if raw_weights else 0.0,
        "norm_weight_min": float(np.min(norm_weights)) if norm_weights else 0.0,
        "norm_weight_max": float(np.max(norm_weights)) if norm_weights else 0.0,
    }
    return summary


def clean_and_prune_circuit(
    graph: nx.DiGraph,
    min_synapses: float = 3.0,
    max_nodes: int = 1000,
    norm_method: Literal["log1p", "minmax", "linear_scale"] = "log1p",
) -> nx.DiGraph:
    """Full pipeline: threshold weak edges, filter dead-ends, normalize, and cap size.
    
    Args:
        graph: Raw extracted NetworkX DiGraph.
        min_synapses: Minimum synapse count threshold.
        max_nodes: Maximum node limit for hardware safety.
        norm_method: Weight normalization method.
        
    Returns:
        Clean, normalized, and bounded DiGraph ready for PyTorch conversion.
    """
    logger.info("Starting graph cleaning pipeline...")
    # 1. Filter weak synapses
    g = filter_by_synapse_threshold(graph, min_synapses=min_synapses)
    # 2. Extract functional pathways (DN -> IN -> MN)
    g = extract_functional_pathways(g, source_role="DN", target_role="MN")
    # 3. Enforce node budget
    g = enforce_node_budget(g, max_nodes=max_nodes)
    # 4. Normalize weights for neural network initialization
    g = normalize_synapse_weights(g, method=norm_method)
    
    logger.info("Graph cleaning pipeline complete.")
    return g


if __name__ == "__main__":
    import pickle
    from pathlib import Path
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    data_dir = Path(__file__).resolve().parents[2] / "data"
    raw_graph_path = data_dir / "dna_motor_circuit.pickle"
    
    if not raw_graph_path.exists():
        print(f"Error: {raw_graph_path} not found. Run fetch_circuit.py first!")
        exit(1)
        
    with open(raw_graph_path, "rb") as f:
        raw_graph = pickle.load(f)
        
    print("\n--- Before Pruning ---")
    raw_summary = get_graph_summary(raw_graph)
    print(f"Nodes: {raw_summary['num_nodes']}, Edges: {raw_summary['num_edges']}")
    print(f"Roles: {raw_summary['roles']}")
    print(f"Raw Synapse Range: [{raw_summary['raw_weight_min']}, {raw_summary['raw_weight_max']}]")
    
    # Run full pruning pipeline
    pruned_graph = clean_and_prune_circuit(
        raw_graph,
        min_synapses=3.0,
        max_nodes=1000,
        norm_method="log1p",
    )
    
    # Save pruned circuit
    pruned_path = data_dir / "dna_motor_circuit_pruned.pickle"
    with open(pruned_path, "wb") as f:
        pickle.dump(pruned_graph, f)
        
    print("\n--- After Pruning & Normalization ---")
    pruned_summary = get_graph_summary(pruned_graph)
    print(f"Nodes: {pruned_summary['num_nodes']}, Edges: {pruned_summary['num_edges']}")
    print(f"Roles: {pruned_summary['roles']}")
    print(f"Normalized Weight Range: [{pruned_summary['norm_weight_min']:.3f}, {pruned_summary['norm_weight_max']:.3f}]")
    print(f"Saved clean graph to: {pruned_path}")
