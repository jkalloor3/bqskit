"""This module implements the MachineModel class."""
from __future__ import annotations

from typing import Sequence
from typing import TYPE_CHECKING

from bqskit.compiler.gateset import GateSet
from bqskit.compiler.gateset import GateSetLike
from bqskit.ir.location import CircuitLocation
from bqskit.ir.gate import Gate
from bqskit.qis.graph import CouplingGraph
from bqskit.qis.graph import CouplingGraphLike
from bqskit.utils.typing import is_integer
from bqskit.utils.typing import is_valid_radixes

import numpy as np

if TYPE_CHECKING:
    from bqskit.ir.circuit import Circuit


DEFAULT_1_QUBIT_LATENCY = 200
DEFAULT_2_QUBIT_LATENCY = 400

class MachineModel:
    """A model of a quantum processing unit."""

    def __init__(
        self,
        num_qudits: int,
        coupling_graph: CouplingGraphLike | None = None,
        gate_set: GateSetLike | None = None,
        radixes: Sequence[int] = [],
        single_qubit_fidelity: float = 0.999,
        two_qubit_fidelity: float = 0.99,
        fidelities: dict[Sequence[int], float] | None = None,
        gate_latencies: dict[Gate, float] | None = None
    ) -> None:
        """
        MachineModel Constructor.

        Args:
            num_qudits (int): The total number of qudits in the machine.

            coupling_graph (Iterable[tuple[int, int]] | None): A coupling
                graph describing which pairs of qudits can interact.
                Given as an undirected edge set. If left as None, then
                an all-to-all coupling graph is used as a default.
                (Default: None)

            gate_set (GateSetLike | None): The native gate set available
                on the machine. If left as None, the default gate set
                will be used. See :func:`~GateSet.default_gate_set`.

            radixes (Sequence[int]): A sequence with its length equal
                to `num_qudits`. Each element specifies the base of a
                qudit. Defaults to qubits.

            single_qubit_fidelity (float): General single qubit fidelity, will 
            be overridden by all values in fidelities

            two_qubit_fidelity (float): General 2-qubit fidelity, is also overriden
            by all values in fidelities
                
            fidelieties (dict[Sequence[int], float]): A mapping from qubit -> fidelity or
            (qubit, qubit) -> 2-qubit gate fidelit for all qubits in the machine. If
            left as none, will construct from 1-q and 2-q fidelity values

        Raises:
            ValueError: If `num_qudits` is nonpositive.

        Note:
            Pre-built models for many active QPUs exist in the `bqskit.ext`
            package.
        """

        if not is_integer(num_qudits):
            raise TypeError(
                f'Expected integer num_qudits, got {type(num_qudits)}.',
            )

        if num_qudits <= 0:
            raise ValueError(f'Expected positive num_qudits, got {num_qudits}.')

        self.radixes = tuple(radixes if len(radixes) > 0 else [2] * num_qudits)

        if not is_valid_radixes(self.radixes):
            raise TypeError('Invalid qudit radixes.')

        if len(self.radixes) != num_qudits:
            raise ValueError(
                'Expected length of radixes to be equal to num_qudits:'
                ' %d != %d' % (len(self.radixes), num_qudits),
            )

        if coupling_graph is None:
            coupling_graph = CouplingGraph.all_to_all(num_qudits)

        if not CouplingGraph.is_valid_coupling_graph(
                coupling_graph, num_qudits,
        ):
            raise TypeError('Invalid coupling graph, expected list of tuples')

        if gate_set is None:
            gate_set = GateSet.default_gate_set(radixes)
        else:
            gate_set = GateSet(gate_set)

        if not isinstance(gate_set, GateSet):
            raise TypeError(f'Expected GateSet, got {type(gate_set)}.')

        self.gate_set = gate_set
        self.coupling_graph = CouplingGraph(coupling_graph)
        self.num_qudits = num_qudits

        # Construct default fidelities
        self.fidelities = {}
        self.single_qubit_fidelity = single_qubit_fidelity
        self.two_qubit_fidelity = two_qubit_fidelity
        for q in range(num_qudits):
            self.fidelities[(q,)] = single_qubit_fidelity

        for edge in coupling_graph:
            self.fidelities[edge] = two_qubit_fidelity

        # Use user inputted fidelities if not none
        if fidelities:
            self.fidelities.update(fidelities)

        if gate_latencies:
            self.gate_latencies = gate_latencies
        else:
            self.gate_latencies = {}
            for gate in self.gate_set:
                self.gate_latencies[gate] = 100

    def get_locations(self, block_size: int) -> list[CircuitLocation]:
        """Return all `block_size` connected blocks of qudit indicies."""
        return self.coupling_graph.get_subgraphs_of_size(block_size)

    def is_compatible(
        self,
        circuit: Circuit,
        placement: list[int] | None = None,
    ) -> bool:
        """Check if a circuit is compatible with this model."""
        if circuit.num_qudits > self.num_qudits:
            return False

        if any(g not in self.gate_set for g in circuit.gate_set):
            return False

        if placement is None:
            placement = list(range(circuit.num_qudits))

        if any(
            (placement[e[0]], placement[e[1]]) not in self.coupling_graph
            for e in circuit.coupling_graph
        ):
            return False

        if any(
            r != self.radixes[placement[i]]
            for i, r in enumerate(circuit.radixes)
        ):
            return False

        return True
    
    def calculate_fidelity(self, circuit: Circuit):
        fid = 1
        for op in circuit:
            if op.gate.name != "barrier":
                if len(op.location) > 2:
                    # Just default to two qubit fidelity for now
                    fid *= self.two_qubit_fidelity
                else:
                    fid *= self.fidelities[tuple(sorted(op.location))]
        return fid
    

    def calculate_cb_fidelity(self, circuit: Circuit, topology_penalty = False):
        fid = 1
        degs = self.coupling_graph.get_qudit_degrees()
        # Convert from avg. fidelity to process fideliy factor
        avg_to_proc = (2 ** circuit.num_qudits + 1) / (2 ** circuit.num_qudits)
        for cycle in range(circuit.num_cycles):
            single_gate_errors = {}
            multi_gate_errors = {}
            for qud in range(circuit.num_qudits):
                try:
                    op = circuit.get_operation((cycle, qud))
                    if op.num_qudits > 1:
                        multi_gate_errors[tuple(sorted(op.location))] = (1 - self.fidelities[tuple(sorted(op.location))]) * (avg_to_proc)
                    else:
                        single_gate_errors[tuple(sorted(op.location))] = (1 - self.fidelities[tuple(sorted(op.location))]) * (avg_to_proc)
                except:
                    continue
            
            # Calculate process infidelity
            single_gate_cycle_fidelity = 1 - sum(single_gate_errors.values())

            # Calculate the process infidelity for the 2-qubit
            total_error = 0
            for gate in multi_gate_errors:
                if topology_penalty:
                    total_error += (degs[gate[0]] + degs[gate[1]]) * multi_gate_errors[gate]
                else:
                    total_error += multi_gate_errors[gate] * len(gate)

            multi_gate_cycle_fidelity = 1 - total_error

            fid *= (single_gate_cycle_fidelity * multi_gate_cycle_fidelity)

        return fid

    def calculate_latency(self, circuit: Circuit):
        latencies = np.zeros(circuit.num_qudits, dtype=int)
        for op in circuit:
            default_latency = DEFAULT_1_QUBIT_LATENCY
            if len(op.location) == 2:
                default_latency = DEFAULT_2_QUBIT_LATENCY
            gate_latency = self.gate_latencies.get(op.gate, default_latency)
            for qubit in op.location:
                latencies[qubit] += gate_latency

        return np.max(latencies)
