"""This module implements the SqrtCNOTGate."""
from __future__ import annotations

from bqskit.ir.gates.constantgate import ConstantGate
from bqskit.ir.gates.qubitgate import QubitGate
from scipy.linalg import eig
from numpy.linalg import inv
import numpy as np
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix


class NRootCNOTGate(ConstantGate, QubitGate):
    """
    The Square root Controlled-X gate.

    The SqrtCNOT gate is given by the following unitary:

    .. math::

        \\begin{pmatrix}
        1 & 0 & 0 & 0 \\\\
        0 & 1 & 0 & 0 \\\\
        0 & 0 & \\frac{1}{2} + \\frac{1}{2}i & \\frac{1}{2} - \\frac{1}{2}i \\\\
        0 & 0 & \\frac{1}{2} - \\frac{1}{2}i & \\frac{1}{2} + \\frac{1}{2}i \\\\
        \\end{pmatrix}
    """

    _num_qudits = 2
    _qasm_name = 'nx'

    def __init__(self, root: int):
        self.root = root
        x = np.array([[0, 1], [1, 0]])
        diag, P = eig(x)
        P_inv = inv(P)
        root_diag = diag ** (1 / root)
        nrootx = P @ np.diag(root_diag) @ P_inv

        self._utry = UnitaryMatrix([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, nrootx[0][0], nrootx[0][1]],
            [0, 0, nrootx[1][0], nrootx[1][1]]
        ])

    @property
    def name(self) -> str:
        """The name of this gate."""
        return f"{self.root}thRootCNOTGate"
