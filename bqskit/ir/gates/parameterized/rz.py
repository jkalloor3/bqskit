"""This module implements the RZGate."""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from bqskit.ir.gates.qubitgate import QubitGate
from bqskit.qis.unitary.differentiable import DifferentiableUnitary
from bqskit.qis.unitary.optimizable import LocallyOptimizableUnitary
from bqskit.qis.unitary.unitary import RealVector
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.cachedclass import CachedClass


class RZGate(QubitGate, DifferentiableUnitary, CachedClass, LocallyOptimizableUnitary):
    """
    A gate representing an arbitrary rotation around the Z axis.

    It is given by the following parameterized unitary:

    .. math::

        \\begin{pmatrix}
        \\exp({i\\frac{\\theta}{2}}) & 0 \\\\
        0 & \\exp({-i\\frac{\\theta}{2}}) \\\\
        \\end{pmatrix}
    """

    _num_qudits = 1
    _num_params = 1
    _qasm_name = 'rz'

    def get_unitary(self, params: RealVector = []) -> UnitaryMatrix:
        """Return the unitary for this gate, see :class:`Unitary` for more."""
        self.check_parameters(params)

        pexp = np.exp(1j * params[0] / 2)
        nexp = np.exp(-1j * params[0] / 2)

        return UnitaryMatrix(
            [
                [nexp, 0],
                [0, pexp],
            ],
        )

    def get_grad(self, params: RealVector = []) -> npt.NDArray[np.complex128]:
        """
        Return the gradient for this gate.

        See :class:`DifferentiableUnitary` for more info.
        """
        self.check_parameters(params)

        dpexp = 1j * np.exp(1j * params[0] / 2) / 2
        dnexp = -1j * np.exp(-1j * params[0] / 2) / 2

        return np.array(
            [
                [
                    [dnexp, 0],
                    [0, dpexp],
                ],
            ], dtype=np.complex128,
        )


    def optimize(self, env_matrix: npt.NDArray[np.complex128]) -> list[float]:
        """
        Return the optimal parameters with respect to an environment matrix.

        See :class:`LocallyOptimizableUnitary` for more info.
        """
        print("Using Python optimization")
        self.check_env_matrix(env_matrix)
        a = np.angle(env_matrix[0, 0])
        b = np.angle(env_matrix[1, 1])
        theta = 2 * np.arccos(a / np.sqrt(a ** 2 + b ** 2))
        theta *= -1 if b > 0 else 1
        return [theta]