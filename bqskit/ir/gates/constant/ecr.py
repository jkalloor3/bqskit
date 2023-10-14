"""This module implements the ECR Gate."""
from __future__ import annotations

from bqskit.ir.gates.constantgate import ConstantGate
from bqskit.ir.gates.qubitgate import QubitGate
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
import math


class ECRGate(ConstantGate, QubitGate):
    """
    The ECR gate as described by https://qiskit.org/documentation/stubs/qiskit.circuit.library.ECRGate.html
    """

    _num_qudits = 2
    _qasm_name = 'cx'
    _utry = UnitaryMatrix(
        [
            [0, 1 / math.sqrt(2), 0,  1j / math.sqrt(2)],
            [1 / math.sqrt(2), 0, -1j / math.sqrt(2), 0],
            [0, 1j / math.sqrt(2), 0, 1/ math.sqrt(2)],
            [-1j / math.sqrt(2), 0, 1 / math.sqrt(2), 0],
        ],
    )
