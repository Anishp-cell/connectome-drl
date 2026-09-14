"""Connectome Circuit Fetcher for MaleCNS v1.0.

Queries the Janelia FlyEM NeuPrint database to extract descending motor
circuits (DN -> VNC Interneurons -> Motor Neurons) for embodied locomotion.
"""

from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd
from neuprint import Client

logger = logging.getLogger(__name__)


class CircuitFetcher:
    """Extracts targeted motor pathways from Janelia NeuPrint (MaleCNS v1.0).
    
    The biological motor hierarchy consists of:
      1. Descending Neurons (DNs): High-level steering and velocity commands from brain.
      2. Interneurons (INs): Coordinating circuits within the Ventral Nerve Cord (VNC).
      3. Motor Neurons (MNs): Output cells directly exciting leg joint muscles.
    """

    DEFAULT_SERVER = "https://neuprint.janelia.org"
    DEFAULT_DATASET = "male-cns:v1.0"

    def __init__(
        self,
        server: str = DEFAULT_SERVER,
        dataset: str = DEFAULT_DATASET,
        token: str | None = None,
    ) -> None:
        """Initialize the NeuPrint client.
        
        Args:
            server: NeuPrint server URL.
            dataset: Dataset identifier (e.g. 'male-cns:v1.0').
            token: API authorization token. If None, checks env var
                   'NEUPRINT_APPLICATION_CREDENTIALS'.
        """
        self.server = server
        self.dataset = dataset
        self.token = token or os.environ.get("NEUPRINT_APPLICATION_CREDENTIALS")

        if not self.token:
            raise ValueError(
                "NeuPrint authorization token is missing. Please provide a token "
                "or set the 'NEUPRINT_APPLICATION_CREDENTIALS' environment variable."
            )

        logger.info(f"Connecting to NeuPrint server '{self.server}' [dataset: {self.dataset}]")
        self.client = Client(self.server, dataset=self.dataset, token=self.token)

    def fetch_descending_neurons(self, dn_types: list[str]) -> pd.DataFrame:
        """Fetch neuron metadata for specified Descending Neuron types.
        
        Args:
            dn_types: List of DN cell types (e.g. ['DNa01', 'DNa02', 'DNg13']).
            
        Returns:
            DataFrame containing bodyId, type, status.
        """
        types_str = ", ".join(f"'{t}'" for t in dn_types)
        query = f"""
        MATCH (dn:Neuron)
        WHERE dn.type IN [{types_str}]
        RETURN dn.bodyId AS bodyId, dn.type AS type, dn.status AS status
        ORDER BY dn.type, dn.bodyId
        """
        df = self.client.fetch_custom(query)
        logger.info(f"Retrieved {len(df)} descending neurons for types {dn_types}")
        return df

    def fetch_dn_downstream_edges(
        self,
        dn_body_ids: list[int],
        min_synapses: int = 3,
    ) -> pd.DataFrame:
        """Fetch all downstream synaptic connections from the target DNs.
        
        Args:
            dn_body_ids: List of source DN body IDs.
            min_synapses: Minimum synapse count threshold to exclude noise.
            
        Returns:
            DataFrame of synaptic edges: source_id, source_type, target_id,
            target_type, and synapse weight.
        """
        ids_str = ", ".join(str(i) for i in dn_body_ids)
        query = f"""
        MATCH (dn:Neuron)-[s:ConnectsTo]->(target:Neuron)
        WHERE dn.bodyId IN [{ids_str}]
          AND s.weight >= {min_synapses}
        RETURN dn.bodyId AS source_id,
               dn.type AS source_type,
               target.bodyId AS target_id,
               target.type AS target_type,
               s.weight AS weight
        ORDER BY s.weight DESC
        """
        df = self.client.fetch_custom(query)
        logger.info(f"Fetched {len(df)} downstream connections from target DNs")
        return df

    def fetch_interneuron_to_mn_edges(
        self,
        interneuron_ids: list[int],
        min_synapses: int = 3,
    ) -> pd.DataFrame:
        """Trace connections from intermediate VNC interneurons to Motor Neurons (MNs).
        
        Args:
            interneuron_ids: Candidate intermediate neuron IDs from step 1.
            min_synapses: Minimum synapse count threshold.
            
        Returns:
            DataFrame of synaptic connections reaching motor neurons.
        """
        if not interneuron_ids:
            return pd.DataFrame(columns=["source_id", "source_type", "target_id", "target_type", "weight"])

        ids_str = ", ".join(str(i) for i in interneuron_ids)
        query = f"""
        MATCH (inter:Neuron)-[s:ConnectsTo]->(mn:Neuron)
        WHERE inter.bodyId IN [{ids_str}]
          AND s.weight >= {min_synapses}
          AND (mn.type CONTAINS 'MN' OR mn.type CONTAINS 'Motor' OR mn.type CONTAINS 'motor')
        RETURN inter.bodyId AS source_id,
               inter.type AS source_type,
               mn.bodyId AS target_id,
               mn.type AS target_type,
               s.weight AS weight
        ORDER BY s.weight DESC
        """
        df = self.client.fetch_custom(query)
        logger.info(f"Fetched {len(df)} connections from VNC interneurons to leg motor neurons")
        return df

    def extract_motor_subgraph(
        self,
        dn_types: list[str] | None = None,
        min_synapses: int = 3,
        max_intermediate_nodes: int = 200,
    ) -> tuple[nx.DiGraph, pd.DataFrame]:
        """Extract a complete 2-hop DN -> Interneuron -> Motor Neuron subgraph.
        
        Args:
            dn_types: Target descending neuron types. Defaults to ['DNa01', 'DNa02'].
            min_synapses: Minimum synapse threshold per edge.
            max_intermediate_nodes: Maximum number of intermediate neurons to include
                                   (keeps graph computationally tractible).
                                   
        Returns:
            tuple of (networkx.DiGraph, neuron_metadata_dataframe)
        """
        if dn_types is None:
            dn_types = ["DNa01", "DNa02"]

        # Step 1: Get source DNs
        dn_df = self.fetch_descending_neurons(dn_types)
        if dn_df.empty:
            raise RuntimeError(f"No neurons found for types: {dn_types}")
            
        dn_ids = dn_df["bodyId"].tolist()

        # Step 2: Get DN -> Interneuron connections
        hop1_edges = self.fetch_dn_downstream_edges(dn_ids, min_synapses=min_synapses)
        if hop1_edges.empty:
            raise RuntimeError(f"No downstream connections found for DN types {dn_types}")

        # Filter intermediate nodes by highest cumulative incoming weight from DNs
        top_intermediates = (
            hop1_edges.groupby("target_id")["weight"]
            .sum()
            .sort_values(ascending=False)
            .head(max_intermediate_nodes)
            .index.tolist()
        )
        filtered_hop1 = hop1_edges[hop1_edges["target_id"].isin(top_intermediates)]

        # Step 3: Get Interneuron -> Motor Neuron connections
        hop2_edges = self.fetch_interneuron_to_mn_edges(
            top_intermediates, min_synapses=min_synapses
        )

        # Step 4: Construct NetworkX Directed Graph
        graph = nx.DiGraph()

        # Add DN nodes
        for _, row in dn_df.iterrows():
            graph.add_node(
                int(row["bodyId"]),
                type=str(row["type"]),
                role="DN",
                status=str(row.get("status", "Traced")),
            )

        # Add Hop 1 edges (DN -> Interneuron)
        for _, row in filtered_hop1.iterrows():
            src = int(row["source_id"])
            tgt = int(row["target_id"])
            tgt_type = str(row["target_type"]) if pd.notna(row["target_type"]) else "Unknown_IN"
            
            if tgt not in graph:
                graph.add_node(tgt, type=tgt_type, role="VNC_IN")
            
            graph.add_edge(src, tgt, weight=float(row["weight"]))

        # Add Hop 2 edges (Interneuron -> MN)
        for _, row in hop2_edges.iterrows():
            src = int(row["source_id"])
            tgt = int(row["target_id"])
            tgt_type = str(row["target_type"]) if pd.notna(row["target_type"]) else "Unknown_MN"
            
            if tgt not in graph:
                graph.add_node(tgt, type=tgt_type, role="MN")
            
            graph.add_edge(src, tgt, weight=float(row["weight"]))

        # Compile metadata DataFrame
        metadata_records = []
        for node, attrs in graph.nodes(data=True):
            metadata_records.append({
                "bodyId": node,
                "type": attrs.get("type", "Unknown"),
                "role": attrs.get("role", "Unknown"),
                "in_degree": graph.in_degree(node),
                "out_degree": graph.out_degree(node),
            })
        metadata_df = pd.DataFrame(metadata_records)

        logger.info(
            f"Constructed motor circuit graph: {graph.number_of_nodes()} nodes, "
            f"{graph.number_of_edges()} edges"
        )
        return graph, metadata_df

    @staticmethod
    def save_circuit(
        graph: nx.DiGraph,
        metadata: pd.DataFrame,
        save_dir: str | Path,
        prefix: str = "motor_subgraph",
    ) -> tuple[Path, Path]:
        """Save graph and metadata to disk for offline caching.
        
        Args:
            graph: NetworkX directed graph.
            metadata: Pandas DataFrame of neuron metadata.
            save_dir: Destination directory.
            prefix: Filename prefix.
            
        Returns:
            Tuple of (graph_path, metadata_path).
        """
        out_path = Path(save_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        graph_file = out_path / f"{prefix}.pickle"
        meta_file = out_path / f"{prefix}_metadata.csv"

        with open(graph_file, "wb") as f:
            pickle.dump(graph, f)
        metadata.to_csv(meta_file, index=False)

        logger.info(f"Saved circuit to {graph_file} and {meta_file}")
        return graph_file, meta_file

    @staticmethod
    def load_circuit(
        graph_path: str | Path,
        metadata_path: str | Path | None = None,
    ) -> tuple[nx.DiGraph, pd.DataFrame | None]:
        """Load cached graph and optional metadata from disk."""
        with open(graph_path, "rb") as f:
            graph = pickle.load(f)
        
        meta_df = None
        if metadata_path is not None and Path(metadata_path).exists():
            meta_df = pd.read_csv(metadata_path)

        return graph, meta_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    token = os.environ.get(
        "NEUPRINT_APPLICATION_CREDENTIALS",
        "a3ad6b2c6b4319d0edde6d62215439f1a4b8e7477e457910100ac2450a1bb7e6",
    )
    
    fetcher = CircuitFetcher(token=token)
    graph, meta = fetcher.extract_motor_subgraph(
        dn_types=["DNa01", "DNa02"],
        min_synapses=3,
        max_intermediate_nodes=150,
    )
    
    # Save to connectome_rl/data/
    save_dir = Path(__file__).resolve().parents[2] / "data"
    graph_path, meta_path = CircuitFetcher.save_circuit(
        graph, meta, save_dir=save_dir, prefix="dna_motor_circuit"
    )
    
    print("\n" + "=" * 50)
    print("MBN/VNC MOTOR CIRCUIT SUMMARY")
    print("=" * 50)
    print(f"Total Neurons (Nodes): {graph.number_of_nodes()}")
    print(f"Total Synaptic Connections (Edges): {graph.number_of_edges()}")
    print("\nNeuron Role Breakdown:")
    print(meta["role"].value_counts().to_string())
    print("\nSample Neurons:")
    print(meta.head(10).to_string(index=False))
    print("\nCached Files:")
    print(f"  Graph:    {graph_path}")
    print(f"  Metadata: {meta_path}")
    print("=" * 50)
