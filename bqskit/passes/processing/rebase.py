"""This module implements the Rebase2QuditGatePass."""
from __future__ import annotations

import logging
from typing import Any, Sequence

from bqskit.ext import bqskit_to_qiskit
from qiskit import QuantumCircuit
from bqskit.ir import Gate
from bqskit.ir.point import CircuitPoint as Point
from bqskit.compiler.basepass import BasePass
from bqskit.ir.circuit import Circuit, Operation
from bqskit.ir.opt.cost.functions import HilbertSchmidtResidualsGenerator
from bqskit.ir.opt.cost.generator import CostFunctionGenerator
from bqskit.utils.typing import is_integer, is_real_number, is_sequence
from bqskit.ir.gates import U3Gate
from bqskit.runtime import get_runtime
from itertools import product
from bqskit.qis.graph import CouplingGraph

_logger = logging.getLogger(__name__)


class RebaseGatePass(BasePass):
    """
    The Rebase2QuditGatePass class.

    Will use instantiation to change the a 2-qudit gate to a different one.
    """

    def __init__(
        self,
        new_gate: Gate | Sequence[Gate],
        max_depth: int = 3,
        max_retries: int = -1,
        success_threshold: float = 1e-10,
        cost: CostFunctionGenerator = HilbertSchmidtResidualsGenerator(),
        instantiate_options: dict[str, Any] = {},
    ) -> None:
        """
        Construct a Rebase2QuditGatePass.

        Args:
            gate_in_circuit (Gate | Sequence[Gate]): The two-qudit gate
                or gates in the circuit that you want to replace.
            
            new_gate (Gate | Sequence[Gate]): The two-qudit new gate or
                gates you want to put in the circuit.
            
            max_depth (int): The maximum number of new gates to replace
                an old gate with. (Default: 3) 
            
            max_retries (int): The number of retries for the same gate
                before we increase the maximum depth. If left as -1,
                then never increase max depth. (Default: -1)

            success_threshold (float): The distance threshold that
                determines successful termintation. Measured in cost
                described by the hilbert schmidt cost function.
                (Default: 1e-10)

            cost (CostFunction | None): The cost function that determines
                successful removal of a gate.
                (Default: HilbertSchmidtResidualsGenerator())

            instantiate_options (dict[str: Any]): Options passed directly
                to circuit.instantiate when instantiating circuit
                templates. (Default: {})
        
        Raises:
            ValueError: If `gate_in_circuit` or `new_gate` is not a 2-qudit
                gate.
            
            ValueError: if `max_depth` is nonnegative.
        """
        if is_sequence(new_gate):
            if any(not isinstance(g, Gate) for g in new_gate):
                raise TypeError(f"Expected Gate or Gate list.")
            
        elif not isinstance(new_gate, Gate):
            raise TypeError(f"Expected Gate, got {type(new_gate)}.")
        else:
            new_gate = [new_gate]

        if not is_integer(max_depth):
            raise TypeError(f"Expected integer, got {max_depth}.")
        
        if max_depth < 0:
            raise ValueError(f"Expected nonnegative depth, got: {max_depth}.")

        if not is_integer(max_retries):
            raise TypeError(f"Expected integer, got {max_retries}.")

        if not is_real_number(success_threshold):
            raise TypeError(
                'Expected real number for success_threshold'
                ', got %s' % type(success_threshold),
            )

        if not isinstance(cost, CostFunctionGenerator):
            raise TypeError(
                'Expected cost to be a CostFunctionGenerator, got %s'
                % type(cost),
            )

        if not isinstance(instantiate_options, dict):
            raise TypeError(
                'Expected dictionary for instantiate_options, got %s.'
                % type(instantiate_options),
            )
        self.ngates = new_gate
        self.max_depth = max_depth
        self.max_retries = max_retries
        self.success_threshold = success_threshold
        self.cost = cost
        self.instantiate_options: dict[str, Any] = {
            'dist_tol': self.success_threshold,
            'min_iters': 100,
        }
        self.instantiate_options.update(instantiate_options)

    async def run(self, circuit: Circuit, data: dict[str, Any] = {}) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""
        self.generate_new_gate_templates(circuit.gate_counts)
        _logger.debug(f'Rebasing gates from {self.gates} to {self.ngates}.')

        target = self.get_target(circuit, data)
        for g in self.gates:
            # Track retries to check for no progress
            num_retries = 0
            prev_count = circuit.count(g)

            while g in circuit.gate_set:
                # Check if we made progress from last loop
                gates_left = circuit.count(g)
                if prev_count == gates_left:
                    num_retries += 1
                else:
                    prev_count = gates_left
                    num_retries = 0

                # Group together a n-qubit block composed of gates from old set
                point, gate_size = self.group_near_gates(circuit, circuit.point(g))
                
                circuits_with_new_gate = []
                for circ in self.circs[gate_size]:
                    circuit_copy = circuit.copy()
                    circuit_copy.replace_with_circuit(point, circ)
                    circuits_with_new_gate.append(circuit_copy)

                # If we have exceeded the number of retries, up the max depth
                if self.max_retries >= 0 and num_retries > self.max_retries:
                    _logger.info("Exceeded max retries, increasing depth.")
                    circuit_copy = circuit.copy()
                    circuit_copy.replace_with_circuit(point, self.overdrives[gate_size])
                    circuits_with_new_gate.append(circuit_copy)
                
                instantiated_circuits = await get_runtime().map(
                    Circuit.instantiate,
                    circuits_with_new_gate,
                    target=target,
                    **self.instantiate_options
                )

                dists = [self.cost(c, target) for c in instantiated_circuits]

                # Find the successful circuit with the least gates
                best_index = None
                best_count = self.max_depth + 2
                for i, dist in enumerate(dists):
                    if dist < self.success_threshold:
                        if self.counts[gate_size][i] < best_count:
                            best_index = i
                            best_count = self.counts[gate_size][i]
                
                if best_index is None:
                    circuit.unfold(point)
                    continue

                _logger.info(self.replaced_log_messages[gate_size][best_index])
                circuit.become(instantiated_circuits[best_index])


    def get_op_pts_at_cycle(self, circuit: Circuit, cycle: int) -> set[Operation]:
        op_pts = set()
        for q in range(circuit.num_qudits):
            pt = (cycle, q)
            if not circuit.is_point_idle(pt):
                op_pts.add(circuit[pt])

        return op_pts

    def group_near_gates(self, circuit: Circuit, center: Point) -> Point:
        """Group gates similar to the gate at center on the same qubits."""
        op = circuit[center]
        used_qubits = set(list(op.location))
        gate_size = len(used_qubits)
        orig_qubits = set(list(op.location))
        max_width = self.max_gate_size

        counts = {g: 0 for g in circuit.gate_set}
        counts[op.gate] = 1

        # Overall algorithm:
        # Start moving left, if you encounter a gate that overlaps with your qubits
        # do the following:
        #   If it can be added and the number of qubits <= max_width:
        #       Add the gate and extend your qubits, note that a gate was added
        #   If it cannot be added, then stop moving left

        # Now with the qubits that were added, check the cycles you have already traversed and current cycle
        # starting from the left. If these gates can be added, add them. If not, revert back to original qubits
        # for right traversal and do not allow the right to add qubits.

        # Now, move right. If allowed to add qubits, do so, otherwise only add if gates fall within qubits.
        # Starting from center, go to left first
        min_cycle = max_cycle = -1
        moving_left = True
        new_qubits = set()
        min_cycle = center.cycle
        i = 1
        visited_pts = set([center])
        while (moving_left):
            if (center.cycle - i < 0):
                moving_left = False
                break
            frontier_ops = self.get_op_pts_at_cycle(circuit, center.cycle - i)
            for op in frontier_ops:
                diff = set(op.location).difference(used_qubits)
                if (len(diff) == 0):
                    op_pts = set(product([center.cycle - i], op.location))
                    visited_pts.update(op_pts)
                    counts[op.gate] += 1
                    min_cycle = center.cycle - i
                elif (len(diff) == len(op.location)):
                    # No overlap
                    continue
                elif ((len(diff) + len(used_qubits)) <= max_width):
                    # Some overlap and can add qubit
                    op_pts = set(product([center.cycle - i], op.location))
                    visited_pts.update(op_pts)
                    counts[op.gate] += 1
                    used_qubits.update(diff)
                    new_qubits.update(diff)
                    min_cycle = center.cycle - i
                else:
                    # Overlap and can't add qubit
                    # Can no longer go left
                    moving_left = False
            i += 1

        right_can_add_qubits = True

        # Check if can add more gates
        pts_to_check = product(range(min_cycle, center.cycle + 1), new_qubits)

        for pt in pts_to_check:
            if not circuit.is_point_idle(pt):
                op = circuit[pt]
                diff = set(op.location).difference(used_qubits)
                if len(diff) == 0:
                    visited_pts.add(pt)
                    counts[op.gate] += 1
                else:
                    # Here, there is overlap! So do not allow right to add qubits
                    # TODO: Right should be able to add other qubits that are not in new qubits
                    right_can_add_qubits = False
                    gate_size = len(used_qubits)
                    used_qubits = orig_qubits
                    break

        # Now move right
        moving_right = True
        new_qubits = set()
        i = 0
        while (moving_right):
            if (center.cycle + i >= circuit.num_cycles):
                moving_right = False
                break
            frontier_ops = self.get_op_pts_at_cycle(circuit, center.cycle + i)
            for op in frontier_ops:
                diff = set(op.location).difference(used_qubits)
                if (len(diff) == 0):
                    op_pts = set(product([center.cycle + i], op.location))
                    visited_pts.update(op_pts)
                    counts[op.gate] += 1
                    max_cycle = center.cycle + i
                elif (len(diff) == len(op.location)):
                    # No overlap
                    continue
                elif ((len(diff) + len(used_qubits)) <= max_width) and right_can_add_qubits:
                    # Some overlap and can add qubit
                    op_pts = set(product([center.cycle + i], op.location))
                    visited_pts.update(op_pts)
                    counts[op.gate] += 1
                    used_qubits.update(diff)
                    new_qubits.update(diff)
                    max_cycle = center.cycle + i
                else:
                    # Overlap and can't add qubit
                    # Can no longer go right
                    max_cycle = center.cycle + i
                    moving_right = False
            i += 1

        # Check if can add more gates
        pts_to_check = product(range(max_cycle, center.cycle - 1, -1), new_qubits)

        for pt in pts_to_check:
            if not circuit.is_point_idle(pt):
                op = circuit[pt]
                diff = set(op.location).difference(used_qubits)
                if len(diff) == 0:
                    visited_pts.add(pt)
                    counts[op.gate] += 1
                else:
                    # Here, there is overlap! So do not allow right to add qubits
                    break

        grouped_gate_str = ", ".join([
            f"{int(c)} {g}" + ("s" if c > 1 else "")
            for g, c in counts.items()
        ])

        # Create region from set of operators
        try:
            region = circuit.get_region(visited_pts)
        except ValueError:
            # qc = bqskit_to_qiskit(circuit)
            # qc.draw(filename="part.txt")
            print(used_qubits)
            print(center)
            print(min_cycle, max_cycle)
            import pickle
            with open("problem_partition.pkl", "wb") as f:
                pickle.dump(circuit, f)
            # print(grouped_gate_str)
            raise ValueError("Problemo")

        _logger.debug(f"Grouped together {grouped_gate_str}.")
        return circuit.fold(region), max(gate_size, len(used_qubits))
    
    def generate_new_gate_templates(self, gate_counts: dict[Gate, int]) -> None:
        gates_in_circuit = [x for x in gate_counts if x.num_qudits >= 2]
        all_gates = gates_in_circuit + self.ngates
        self.max_gate_size = max(q.num_qudits for q in all_gates)
        self.gates = gates_in_circuit
        
        """Generate the templates to be instantiated over old circuits."""
        self.circs: dict[int, list[Circuit]] = {}
        self.counts: dict[int, list[int]] = {}
        # Make circuits for all sizes 2 -> max gate size
        for i in range(2, self.max_gate_size + 1):
            circ = Circuit(i)
            for j in range(i):
                circ.append_gate(U3Gate(), j)
        
            self.circs[i] = [circ]
            self.counts[i] = [0]


        for i in self.circs:
            # Assume all to all
            # TODO: Take in topology if needed
            cg = CouplingGraph.all_to_all(i)
            
            # TODO: Need better mixing
            # Algorithm: For each depth, add n layers, one for each type of gate
            # Each layer is a full covering of the qubits (probably unecessary)
            for _ in range(1, self.max_depth + 1):
                for g in self.ngates:
                    if (g.num_qudits > i):
                        continue
                    prev_circ = self.circs[i][-1].copy()
                    for loc in cg.get_subgraphs_of_size(g.num_qudits):
                        prev_circ.append_gate(g, loc)
                        for j in loc:
                            prev_circ.append_gate(U3Gate(), j)
                        next_circ = prev_circ.copy()
                        self.circs[i].append(prev_circ)
                        self.counts[i].append(self.counts[i][-1] + 1)
                        prev_circ = next_circ
            
        # Add overdrive circuit, incase we exceed retry limit
        self.overdrives = dict([(i, self.circs[i][-1].copy()) for i in range(2, self.max_gate_size + 1)])
        for i in self.overdrives:
            cg = CouplingGraph.all_to_all(i)
            oc_count = 0
            for g in self.ngates:
                if (g.num_qudits > i):
                    continue
                for loc in cg.get_subgraphs_of_size(g.num_qudits):
                    self.overdrives[i].append_gate(g, loc)
                    for j in loc:
                        self.overdrives[i].append_gate(U3Gate(), j)
                    oc_count += 1
            self.counts[i].append(self.counts[i][-1] + oc_count)
        
        # Preprocess log messages
        self.replaced_log_messages = dict([(i, []) for i in self.circs])
        for i in self.circs:
            for circ in self.circs[i] + [self.overdrives[i]]:
                counts = [circ.count(g) for g in circ.gate_set]
                gate_count_str = ', '.join([
                    f'{c} {g}' + ('s' if c > 1 else '')
                    for c, g in zip(counts, circ.gate_set)
                ])
                msg = f'Replaced gate with {gate_count_str}.'
                self.replaced_log_messages[i].append(msg)