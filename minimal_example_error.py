"""This example script demonstrates calling the different variants of PAM."""
from __future__ import annotations

from bqskit.compiler import Compiler
from bqskit.compiler import GateSet
from bqskit.compiler.compile import build_seqpam_mapping_optimization_workflow
from bqskit.compiler import MachineModel
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import *
from bqskit.passes import *
from bqskit.qis import CouplingGraph
import numpy as np
import pickle
from bqskit.qis.permutation import PermutationMatrix


def mesh_model(n_rows, n_cols, gate_set):
    return MachineModel(n_cols * n_rows, coupling_graph=CouplingGraph.grid(n_rows, n_cols), gate_set=gate_set)

gate_names = {
    "xx": XXGate(),
    "zz": ZZGate(),
    "syc": SycamoreGate(),
    "sqisw": SqrtISwapGate(),
    "cz": CZGate(),
    "cx": CNOTGate(),
    "cnot": CNOTGate(),
    "b": BGate(),
    # "sq4cnot": NRootCNOTGate(4),
    # "sq8cnot": NRootCNOTGate(8)
}

layer_generator = lambda x: SimpleLayerGenerator(two_qudit_gate=x)
synthesis_pass = lambda x: QSearchSynthesisPass(layer_generator=layer_generator(x))


if __name__ == '__main__':

    # num_qubits = 7
    # circuit = Circuit(num_qubits)
    # all_qubits = list(range(num_qubits))

    # for i in range(num_qubits):
    #     qubits = np.random.choice(all_qubits, 2, replace=False)
    #     circuit.append_gate(BGate(), tuple(qubits))
    #     circuit.append_gate(U3Gate(), qubits[0], np.random.random(3) * np.pi)

    circuit = pickle.load(open("min_example_err2.pkl", "rb"))

    # pickle.dump(circuit, open("min_example_error.pkl", "wb"))

    print('Initial Gate Counts:', circuit.gate_counts)
    print(circuit.num_qudits)

    compiler = Compiler()

    gate_set = GateSet([CZGate(), U3Gate()])

    machine_model = mesh_model(3, 2, gate_set)

    print(machine_model.num_qudits)
    print(machine_model.gate_set)

    # Embed into larger circuit
    new_circuit = Circuit(machine_model.num_qudits)
    new_circuit.append_circuit(circuit=circuit, location=tuple(range(circuit.num_qudits)))
    circuit.become(new_circuit)

    print(circuit.num_operations)
    print(circuit.num_qudits)

    workflow = [
            SetModelPass(machine_model),
            build_seqpam_mapping_optimization_workflow(4, synthesis_epsilon=1e-10, error_sim_size=8),
    ]

    out_circuit, data = compiler.compile(circuit, workflow=workflow, request_data=True)
    print('Final Gate Counts:', out_circuit.gate_counts)
    upper_bound_error = data.error

    print(f"Upper Bound Error: {upper_bound_error}")

    # Calculate actual error

    un1 = circuit.get_unitary()
    pi = data['initial_mapping']
    pf = data['final_mapping']

    PI = PermutationMatrix.from_qubit_location(out_circuit.num_qudits, pi)
    PF = PermutationMatrix.from_qubit_location(out_circuit.num_qudits, pf)

    un2 = out_circuit.get_unitary()

    dist = un1.get_distance_from(PF @ un2 @ PI.T)

    print(f"Actual Error: {dist}")