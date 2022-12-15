
from typing import Any
from bqskit import Circuit
from bqskit.compiler.basepass import BasePass
from bqskit.ir.gates.parameterized.pauli import PauliGate
from bqskit.ir.gates.parameterized.unitary import VariableUnitaryGate
from bqskit.passes.search.generator import LayerGenerator
from bqskit.qis.state.state import StateVector
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix


class TwoUnitaryLayerGen(LayerGenerator):

    def gen_initial_layer(self, target: UnitaryMatrix | StateVector, data: dict[str, Any]) -> Circuit:
        cir = Circuit(target.num_qudits, target.radixes)
        
        for i in range(target.num_qudits):
            cir.append_gate(VariableUnitaryGate(num_qudits=1, radixes=[target.radixes[i]]), i)
            break
        
        return cir
    
    def gen_successors(self, circuit: Circuit, data: dict[str, Any]) -> list[Circuit]:
        # Get the coupling graph
        coupling_graph = BasePass.get_connectivity(circuit, data)

        # Generate successors
        successors = []
        for edge in coupling_graph:
            successor = circuit.copy()
            successor.append_gate(VariableUnitaryGate(num_qudits=2, radixes=[circuit.radixes[i] for i in edge]), edge)
            successors.append(successor)

        return successors
    


class TwoUnitaryLayerGenCeres(LayerGenerator):

    def gen_initial_layer(self, target: UnitaryMatrix | StateVector, data: dict[str, Any]) -> Circuit:
        cir = Circuit(target.num_qudits, target.radixes)
        
        for i in range(target.num_qudits):
            cir.append_gate(PauliGate(1), i)

        return cir
    
    def gen_successors(self, circuit: Circuit, data: dict[str, Any]) -> list[Circuit]:
        # Get the coupling graph
        coupling_graph = BasePass.get_connectivity(circuit, data)

        # Generate successors
        successors = []
        for edge in coupling_graph:
            successor = circuit.copy()
            successor.append_gate(PauliGate(2), edge)
            successors.append(successor)

        return successors