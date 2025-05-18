"""This module implements the QFTGate."""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from bqskit.ir.gates.qubitgate import QubitGate
from bqskit.qis.unitary.differentiable import DifferentiableUnitary
from bqskit.qis.unitary.optimizable import LocallyOptimizableUnitary
from bqskit.qis.unitary.unitary import RealVector
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.cachedclass import CachedClass

class QFTGate(
    QubitGate,
    CachedClass,
):
    """
    A gate representing a QFT on n qubits. Can be approximated with 
    the approx parameter. If the approx parameter is small, it takes
    a very far QFT approximation. If the approx parameter is large, it
    will be closer.
    """
    _num_params = 0

    def __init__(self, num_qudits: int, inv: bool = False, approx: int = -1) -> None:
        self._num_qudits = num_qudits
        self.approx = approx
        self.inv = inv
        if inv:
            self._qasm_name = 'iqft'
        else:
            self._qasm_name = 'qft'


    def get_unitary(self, params: RealVector = []) -> UnitaryMatrix:
        """Return the unitary for this gate, see :class:`Unitary` for more."""
        self.check_parameters(params)

        mat = np.zeros((2 ** self.num_qudits, 2 ** self.num_qudits), dtype=np.complex128)

        inv_factor = -1 if self.inv else 1

        if self.approx > 0:
            assert self.approx <= self.num_qudits
            # Create some unitary
            for i in range(2 ** self.num_qudits):
                a = [int(x) for x in bin(i)[2:]]
                a.reverse()
                for j in range(2 ** self.num_qudits):
                    c = [int(x) for x in bin(j)[2:]]
                    c.reverse()
                    exp_sum = 0
                    for ind_1 in range(len(a)):
                        for ind_2 in range(len(c)):
                            ind_sum = ind_1 + ind_2
                            if ind_sum < (self.num_qudits) and ind_sum >= (self.num_qudits - self.approx):
                                exp_sum += a[ind_1] * c[ind_2] * 2 ** (ind_1 + ind_2)
                    
                    exp_sum *= inv_factor * 2 * np.pi * 1j / (2 ** self.num_qudits)
                    mat[i, j] = np.exp(exp_sum) / (2 ** (self.num_qudits / 2))
        else:
            # Exact QFT unitary
            w = np.exp(2 * np.pi * 1j / (2 ** self.num_qudits))
            for i in range(2 ** self.num_qudits):
                for j in range(2 ** self.num_qudits):
                    
                    mat[i, j] = w ** (i * j * inv_factor) / (2 ** (self.num_qudits / 2))

        return UnitaryMatrix(
            mat
        )