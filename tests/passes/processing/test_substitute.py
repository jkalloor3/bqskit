from __future__ import annotations

from bqskit.compiler import CompilationTask
from bqskit.compiler import Compiler
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import VariableUnitaryGate
from bqskit.ir.operation import Operation
from bqskit.passes import SubstitutePass
from bqskit.qis import UnitaryMatrix


class TestSubstitute:

    def test_small_qubit(self) -> None:
        utry = UnitaryMatrix.identity(4)
        circuit = Circuit(2)
        circuit.append_gate(VariableUnitaryGate(2), [0, 1])
        circuit.instantiate(utry)
        assert circuit.get_unitary().get_distance_from(utry) < 1e-5

        def is_variable(op: Operation) -> bool:
            return isinstance(op.gate, VariableUnitaryGate)
        substitute = SubstitutePass(is_variable, VariableUnitaryGate(1))
        substitute.run(circuit)
        dist = circuit.get_unitary().get_distance_from(utry)
        assert dist <= 1e-5
        assert circuit.num_operations == 1
        assert circuit[0, 0].num_qudits == 1

    def test_small_qubit_with_compiler(self, compiler: Compiler) -> None:
        utry = UnitaryMatrix.identity(4)
        circuit = Circuit(2)
        circuit.append_gate(VariableUnitaryGate(2), [0, 1])
        circuit.instantiate(utry)
        assert circuit.get_unitary().get_distance_from(utry) < 1e-5

        def is_variable(op: Operation) -> bool:
            return isinstance(op.gate, VariableUnitaryGate)
        substitute = SubstitutePass(is_variable, VariableUnitaryGate(1))
        circuit = compiler.compile(CompilationTask(circuit, [substitute]))
        dist = circuit.get_unitary().get_distance_from(utry)
        assert dist <= 1e-5
        assert circuit.num_operations == 1
        assert circuit[0, 0].num_qudits == 1
