"""This example script demonstrates calling the different variants of PAM."""
from __future__ import annotations

from bqskit.compiler import Compiler, compile
from bqskit.compiler import GateSet
from bqskit.compiler import MachineModel
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import *
from bqskit.passes import *
from bqskit.qis import CouplingGraph
import numpy as np
import pickle


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
    circuit = pickle.load(open("min_example.pkl", "rb"))

    print('Initial Gate Counts:', circuit.gate_counts)
    print(circuit.num_qudits)

    compiler = Compiler()

    gate_set = GateSet([BGate(), U3Gate()])

    machine_model = mesh_model(3, 3, gate_set)

    print(machine_model.num_qudits)

    print(machine_model.gate_set)

    out_circuit = compile(circuit, machine_model, optimization_level=4, compiler=compiler)

    print('Final Gate Counts:', out_circuit.gate_counts)