"""This module implements the OPENQASM2Language class."""
from __future__ import annotations

from typing import TYPE_CHECKING

from bqskit.ir.lang.language import LangException
from bqskit.ir.lang.language import Language
from bqskit.ir.lang.qasm2.parser import parse
from bqskit.ir.lang.qasm2.visitor import OPENQASMVisitor, GateDef

if TYPE_CHECKING:
    from bqskit.ir.circuit import Circuit


class OPENQASM2Language(Language):
    """The OPENQASM2Language class."""
    def __init__(self, gate_defs: list[tuple[str, GateDef]] = []) -> None:
        self.gate_defs = {}
        for gate_def in gate_defs:
            self.gate_defs[gate_def[0]] = gate_def[1]

    def encode(self, circuit: Circuit) -> str:
        """Write `circuit` in this language."""
        if not circuit.is_qubit_only():
            raise LangException('Only qubit circuits can be wrriten to qasm.')

        source = "OPENQASM 2.0;\ninclude \"qelib1.inc\";\n"
        source += f'qreg q[{circuit.num_qudits}];\n'
        for gate in circuit.gate_set:
            source += gate.get_qasm_gate_def()

        for op in circuit:
            source += op.get_qasm()

        return source

    def decode(self, source: str, gate_defs: list[tuple[str, GateDef]] = []) -> Circuit:
        """Parse `source` into a circuit."""
        tree = parse(source)
        visitor = OPENQASMVisitor()
        # Add gate defs to visitor
        for gate_def in self.gate_defs.items():
            visitor.gate_defs[gate_def[0]] = gate_def[1]
        for gate_def in gate_defs:
            visitor.gate_defs[gate_def[0]] = gate_def[1]
        visitor.visit_topdown(tree)
        return visitor.get_circuit()
