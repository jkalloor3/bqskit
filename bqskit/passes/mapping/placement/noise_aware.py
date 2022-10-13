"""This module implements the GreedyPlacementPass class."""
from __future__ import annotations

import logging
from typing import Any, OrderedDict

import numpy as np

from bqskit.compiler.basepass import BasePass
from bqskit.ir.circuit import Circuit, CircuitLocation
from bqskit.qis.graph import CouplingGraph

import itertools as it

_logger = logging.getLogger(__name__)


class NoiseAwarePlacementPass(BasePass):
    """Find a placement by starting with the most connected qudit."""

    def run(self, circuit: Circuit, data: dict[str, Any] = {}) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""

        # Find all subgraphs of size circuit_width in the model
        machine_model = self.get_model(circuit, data)
        noise_model = machine_model.qudit_noise_model
        coupling_graph = machine_model.coupling_graph

        subgraph_options = machine_model.get_locations(circuit.num_qudits)

        best_score = 0
        best_subgraph = None

        # For each subgraph, create a score based on 
        for loc in subgraph_options:
            graph = coupling_graph.get_induced_subgraph(loc)
            noise_score = self.get_noise_score(graph, noise_model)
            connectivity_score = self.get_connectivity_score(graph, circuit)
            total_score = self.get_total_score(graph, circuit, noise_score, connectivity_score)

            if total_score > best_score:
                best_score = total_score
                best_subgraph = loc

        # Set locations as best placement
        placement = [i for i in best_subgraph]

        print(placement)

        data['placement'] = sorted(placement)

        _logger.info(f'Placed qudits on {data["placement"]}')

        # Raise an error if this is not a valid placement
        sg = coupling_graph.get_subgraph(data['placement'])
        if not sg.is_fully_connected():
            raise RuntimeError('No valid placement found.')


    def get_total_score(self, subgraph: CouplingGraph, circuit: Circuit, noise_score: float, conn_score: float):
        # Combine noise and conn score. With future experiments, we will probably have to consider depth of the circuit and other details
        return noise_score * conn_score


    # Get the score of the subgraph chosen in the model, compared to the subgraph as defined by the circuit
    # 1 means every edge exists in the subgraph, so the connectivity is as good as needed for the circuit
    def get_connectivity_score(self, graph_edges: list[tuple(int, int)], circuit: Circuit):
        subgraph = CouplingGraph(graph_edges)
        dist_mat = subgraph.all_pairs_shortest_path()
        circuit_coupling_graph = circuit.coupling_graph
        # Want to check how isomorphic two graphs are!
        # We can order vertices by degrees as heuristic for 
        # matching, and then add penalties for missing edges
        circuit_vertices = sorted([(d,i) for (i,d) in enumerate(circuit_coupling_graph.get_qudit_degrees())], reverse=True)
        subgraph_vertices = sorted([(d,i) for (i,d) in enumerate(subgraph.get_qudit_degrees())], reverse=True)

        circuit_to_sugraph_mapping = {}
        for i,pair in enumerate(circuit_vertices):
            circ_vertex = pair[1]
            # Get vertex corresponding to same index
            circuit_to_sugraph_mapping[circ_vertex] = subgraph_vertices[i][1] 

        penalty = 0.0
        num_edges = len(circuit_coupling_graph._edges)
        for (q1, q2) in circuit_coupling_graph._edges:
            subgraph_q1 = circuit_to_sugraph_mapping[q1]
            subgraph_q2 = circuit_to_sugraph_mapping[q2]
            penalty += dist_mat[subgraph_q1][subgraph_q2]

        return num_edges / penalty


    # Fairly simple equation, simply look at the fidelity of each qudit chosen in the subgraph
    def get_noise_score(self, graph: list[tuple(int, int)], qudit_error_model: list[float]):
        noise = 0.0
        all_qubits = set((sum(graph, ())))

        for q in all_qubits:
            noise += qudit_error_model[q]
        return noise / len(all_qubits)