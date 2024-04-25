"""This module implements the MCRZGate."""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from bqskit.ir.gates.qubitgate import QubitGate
from bqskit.qis.unitary.differentiable import DifferentiableUnitary
from bqskit.qis.unitary.optimizable import LocallyOptimizableUnitary
from bqskit.qis.unitary.unitary import RealVector
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.cachedclass import CachedClass
from bqskit.ir.gates.parameterized.mcry import get_indices
from typing import Any

class MCRZGate(
    QubitGate,
    DifferentiableUnitary,
    CachedClass,
    LocallyOptimizableUnitary
):
    """
    A gate representing a multiplexed Z rotation.

    It is given by the following parameterized unitary:
    """
    _qasm_name = 'mcrz'

    def __init__(self, num_qudits: int, controlled_qubit: int) -> None:
        self._num_qudits = num_qudits
        # 1 param for each configuration of the selec qubits
        self._num_params = 2 ** (num_qudits - 1)
        self.controlled_qubit = controlled_qubit
        super().__init__()

    def get_unitary(self, params: RealVector = []) -> UnitaryMatrix:
        """Return the unitary for this gate, see :class:`Unitary` for more."""
        if len(params) == 1:
            # If we want to turn on only if all the selects are 1, that corresponds to
            # [0, 0, ...., 0, theta] so that is why we pad in front
            params = list(np.zeros(len(params) - self.num_params)) + list(params)
        self.check_parameters(params)

        matrix = np.zeros((2 ** self.num_qudits, 2 ** self.num_qudits), dtype=np.complex128)
        for i, param in enumerate(params):
            pos = np.exp(1j * param / 2)
            neg = np.exp(-1j * param / 2)

            # Now, get indices based on control qubit.
            # i corresponds to the configuration of the 
            # select qubits (e.g 5 = 101). Now, the 
            # controlled qubit is 0,1 for both the row and col
            # indices. So, if i = 5 and the controlled_qubit is 2
            # Then the rows/cols are 1001 and 1101
            # Use helper function
            x1, x2 = get_indices(i, self.controlled_qubit, self.num_qudits)

            matrix[x1, x1] = neg
            matrix[x2, x2] = pos

        return UnitaryMatrix(matrix)

    def get_grad(self, params: RealVector = []) -> npt.NDArray[np.complex128]:
        """
        Return the gradient for this gate.

        See :class:`DifferentiableUnitary` for more info.
        """
        if len(params) < self.num_params:
            # Pad Zeros to front
            params = list(np.zeros(len(params) - self.num_params)) + list(params)
        self.check_parameters(params)

        matrix = np.zeros((2 ** self.num_qudits, 2 ** self.num_qudits), dtype=np.complex128)
        for i, param in enumerate(params):
            dpos = 1j / 2 * np.exp(1j * param / 2)
            dneg = -1j / 2 * np.exp(-1j * param / 2)

            x1, x2 = get_indices(i, self.controlled_qubit, self.num_qudits)

            matrix[x1, x1] = dpos
            matrix[x2, x2] = dneg

        return UnitaryMatrix(matrix)
    

    def optimize(self, env_matrix: npt.NDArray[np.complex128]) -> list[float]:
        """
        Return the optimal parameters with respect to an environment matrix.

        See :class:`LocallyOptimizableUnitary` for more info.
        """
        self.check_env_matrix(env_matrix)
        thetas = [0] * self.num_params

        for i in range(self.num_params):
            x1, x2 = get_indices(i, self.controlled_qubit, self.num_qudits)
            # Optimize each RZ independently from indices
            # Taken from QFACTOR repo
            a = np.angle(env_matrix[x1, x1])
            b = np.angle(env_matrix[x2, x2])
            # print(thetas)
            thetas[i] = a - b
            
        return thetas
    
    @property
    def name(self) -> str:
        """The name of this gate."""
        base_name = getattr(self, '_name', self.__class__.__name__)
        return f"{base_name}_{self.num_qudits}"