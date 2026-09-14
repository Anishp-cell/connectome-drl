"""Unit tests for Phase 1 Connectome Ingestion, Pruning, and Graph Utilities."""

from pathlib import Path
import pytest
import networkx as nx
import numpy as np
import torch

from connectome_rl.src.connectome.prune import (
    filter_by_synapse_threshold,
    remove_orphan_nodes,
    extract_functional_pathways,
    normalize_synapse_weights,
    enforce_node_budget,
    clean_and_prune_circuit,
)
from connectome_rl.src.connectome.graph_utils import (
    build_node_mappings,
    graph_to_full_adjacency,
    graph_to_hierarchical_masks,
    graph_to_edge_index,
    convert_graph_to_circuit_data,
    load_circuit_data,
    ConnectomeCircuitData,
)


@pytest.fixture
def sample_motor_graph() -> nx.DiGraph:
    """Create a minimal synthetic connectome graph for testing.
    
    Structure:
      DN1 (bodyId=1)  -> IN1 (bodyId=10) with weight=10.0
      DN2 (bodyId=2)  -> IN1 (bodyId=10) with weight=5.0
      DN2 (bodyId=2)  -> IN2 (bodyId=20) with weight=15.0
      IN1 (bodyId=10) -> MN1 (bodyId=100) with weight=8.0
      IN2 (bodyId=20) -> MN2 (bodyId=200) with weight=12.0
      
      Noise/Dead-end elements:
      DN1 (bodyId=1)  -> IN_dead (bodyId=30) with weight=1.0  (weak synapse < 3.0)
      DN1 (bodyId=1)  -> IN_orphan (bodyId=40) with weight=5.0 (dead-end: never reaches MN)
    """
    g = nx.DiGraph()
    # Add nodes with roles
    g.add_node(1, type="DNa01", role="DN")
    g.add_node(2, type="DNa02", role="DN")
    
    g.add_node(10, type="IN_test1", role="VNC_IN")
    g.add_node(20, type="IN_test2", role="VNC_IN")
    g.add_node(30, type="IN_weak", role="VNC_IN")
    g.add_node(40, type="IN_dead", role="VNC_IN")
    
    g.add_node(100, type="Ti_flexor_MN", role="MN")
    g.add_node(200, type="Ti_extensor_MN", role="MN")
    
    # Add functional edges
    g.add_edge(1, 10, weight=10.0)
    g.add_edge(2, 10, weight=5.0)
    g.add_edge(2, 20, weight=15.0)
    g.add_edge(10, 100, weight=8.0)
    g.add_edge(20, 200, weight=12.0)
    
    # Add noise edges
    g.add_edge(1, 30, weight=1.0)   # Below threshold
    g.add_edge(1, 40, weight=5.0)   # Dead end (no path to MN)
    
    return g


def test_filter_by_synapse_threshold(sample_motor_graph):
    """Test that edges with weight < 3.0 are pruned."""
    pruned = filter_by_synapse_threshold(sample_motor_graph, min_synapses=3.0)
    assert not pruned.has_edge(1, 30), "Weak edge (weight=1.0) should have been pruned"
    assert pruned.has_edge(1, 10), "Strong edge (weight=10.0) should be retained"
    # Node 30 should be removed as it became isolated
    assert 30 not in pruned


def test_extract_functional_pathways(sample_motor_graph):
    """Test that dead-end neurons (unable to reach MN) are removed."""
    pruned = extract_functional_pathways(sample_motor_graph, source_role="DN", target_role="MN")
    # IN_dead (node 40) reaches no MN, so it should be removed
    assert 40 not in pruned, "Dead-end neuron 40 should be removed"
    # Functional nodes must be preserved
    assert 1 in pruned and 2 in pruned
    assert 10 in pruned and 20 in pruned
    assert 100 in pruned and 200 in pruned


def test_normalize_synapse_weights(sample_motor_graph):
    """Test log1p normalization of synaptic weights."""
    norm_g = normalize_synapse_weights(sample_motor_graph, method="log1p")
    w_raw = sample_motor_graph[1][10]["weight"]  # 10.0
    w_norm = norm_g[1][10]["norm_weight"]
    expected = np.log1p(w_raw)
    assert np.isclose(w_norm, expected, atol=1e-5)


def test_enforce_node_budget(sample_motor_graph):
    """Test capping graph size while protecting DNs and MNs."""
    capped_g = enforce_node_budget(sample_motor_graph, max_nodes=5)
    assert capped_g.number_of_nodes() <= 5
    # All DNs and MNs should be prioritized
    dns = [n for n, d in capped_g.nodes(data=True) if d.get("role") == "DN"]
    assert len(dns) == 2, "All DNs must be preserved"


def test_build_node_mappings(sample_motor_graph):
    """Test contiguous zero-indexing and ordering of nodes."""
    node_to_idx, idx_to_node, role_indices = build_node_mappings(sample_motor_graph)
    assert len(node_to_idx) == sample_motor_graph.number_of_nodes()
    # Indices must be contiguous 0, 1, ..., N-1
    assert sorted(node_to_idx.values()) == list(range(len(node_to_idx)))
    # DNs must come first
    assert role_indices["DN"] == [0, 1]


def test_hierarchical_masks(sample_motor_graph):
    """Test bipartite mask creation for feedforward layers."""
    # First clean graph so only clean 2-hop paths exist
    clean_g = clean_and_prune_circuit(sample_motor_graph, min_synapses=3.0)
    node_to_idx, _, role_indices = build_node_mappings(clean_g)
    
    hop1_mask, hop1_weights, hop2_mask, hop2_weights = graph_to_hierarchical_masks(
        clean_g, node_to_idx, role_indices, weight_attr="norm_weight"
    )
    
    num_dn = len(role_indices["DN"])
    num_in = len(role_indices["VNC_IN"])
    num_mn = len(role_indices["MN"])
    
    assert hop1_mask.shape == (num_in, num_dn)
    assert hop2_mask.shape == (num_mn, num_in)
    assert hop1_weights.shape == (num_in, num_dn)
    assert hop2_weights.shape == (num_mn, num_in)
    
    # Verify values: DN1 (col 0) -> IN1 (row 0) should be True
    assert hop1_mask.sum().item() > 0
    assert hop2_mask.sum().item() > 0
    assert not torch.isnan(hop1_weights).any()
    assert not torch.isnan(hop2_weights).any()


def test_cached_real_circuit_data():
    """Verify that the real cached MaleCNS circuit data loads correctly and satisfies specs."""
    data_path = Path(__file__).resolve().parents[1] / "data" / "dna_circuit_tensors.pt"
    if not data_path.exists():
        pytest.skip("dna_circuit_tensors.pt not found on disk (run graph_utils.py first)")
        
    circuit = load_circuit_data(data_path)
    assert isinstance(circuit, ConnectomeCircuitData)
    
    # Check node counts
    assert circuit.num_nodes == 506
    assert len(circuit.dn_indices) == 4
    assert len(circuit.in_indices) == 125
    assert len(circuit.mn_indices) == 377
    assert circuit.num_nodes < 1000, "Must satisfy <1000 node hardware limit"
    
    # Check tensor shapes
    assert circuit.hop1_mask.shape == (125, 4)
    assert circuit.hop2_mask.shape == (377, 125)
    assert circuit.full_mask.shape == (506, 506)
    assert circuit.edge_index.shape[0] == 2
    assert circuit.edge_index.shape[1] == 1581
    
    # Check that weights are positive and non-NaN
    assert not torch.isnan(circuit.hop1_weights).any()
    assert not torch.isnan(circuit.hop2_weights).any()
    assert not torch.isinf(circuit.full_weights).any()
    assert circuit.hop1_weights.max() > 0.0
    assert circuit.hop2_weights.max() > 0.0
