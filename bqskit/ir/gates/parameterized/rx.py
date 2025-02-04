"""This module implements the RXGate."""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from bqskit.ir.gates.qubitgate import QubitGate
from bqskit.qis.unitary.differentiable import DifferentiableUnitary
from bqskit.qis.unitary.optimizable import LocallyOptimizableUnitary
from bqskit.qis.unitary.unitary import RealVector
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.cachedclass import CachedClass


class RXGate(
    QubitGate,
    DifferentiableUnitary,
    LocallyOptimizableUnitary,
    CachedClass,
):
    """
    A gate representing an arbitrary rotation around the X axis.

    It is given by the following parameterized unitary:

    .. math::

        \\begin{pmatrix}
        \\cos{\\frac{\\theta}{2}} & -\\sin{\\frac{\\theta}{2}}i \\\\
        -\\sin{\\frac{\\theta}{2}}i & \\cos{\\frac{\\theta}{2}} \\\\
        \\end{pmatrix}
    """

    _num_qudits = 1
    _num_params = 1
    _qasm_name = 'rx'

    def get_unitary(self, params: RealVector = []) -> UnitaryMatrix:
        """Return the unitary for this gate, see :class:`Unitary` for more."""
        self.check_parameters(params)

        cos = np.cos(params[0] / 2)
        sin = -1j * np.sin(params[0] / 2)

        return UnitaryMatrix(
            [
                [cos, sin],
                [sin, cos],
            ],
        )

    def get_grad(self, params: RealVector = []) -> npt.NDArray[np.complex128]:
        """
        Return the gradient for this gate.

        See :class:`DifferentiableUnitary` for more info.
        """
        self.check_parameters(params)

        dcos = -np.sin(params[0] / 2) / 2
        dsin = -1j * np.cos(params[0] / 2) / 2

        return np.array(
            [
                [
                    [dcos, dsin],
                    [dsin, dcos],
                ],
            ], dtype=np.complex128,
        )

    def optimize(self, env_matrix: npt.NDArray[np.complex128]) -> list[float]:
        """
        Return the optimal parameters with respect to an environment matrix.

        See :class:`LocallyOptimizableUnitary` for more info.
        """
        self.check_env_matrix(env_matrix)
        a = np.real(env_matrix[0, 0] + env_matrix[1, 1])
        b = np.imag(env_matrix[0, 1] + env_matrix[1, 0])
        theta = 2 * np.arccos(a / np.sqrt(a ** 2 + b ** 2))
        theta *= -1 if b < 0 else 1
        return [theta]


    @staticmethod
    def is_rx(unitary: UnitaryMatrix, verbose: bool = False) -> float:
        
        top_left = unitary[0, 0]
        if np.allclose(top_left, 0 + 0j):
            # Assert that off-diagonal elements are 1j or -1j
            a = np.allclose(unitary[0, 1], 1j) or np.allclose(unitary[0, 1], -1j)
            b = np.allclose(unitary[1, 0], unitary[0, 1])
            return a and b
        else:
            # global phase removes complex part of top-left
            global_phase = top_left / np.abs(top_left)
            # Calculate potential thetas
            theta = np.arccos(np.abs(top_left)) * 2
            # Check rest of unitary
            cos = np.cos(theta / 2) * global_phase
            sin_1 = -1j * np.sin(theta / 2) * global_phase
            sin_2 = -1 * sin_1 # Same theta could be represented by -sin
            if verbose:
                print("Global Phase: ", global_phase)
                print("Theta: ", theta)
                print("Cos: ", cos)
                print("Sin: ", sin_1)
                print("Sin_2: ", sin_2)
            top_right = np.allclose(unitary[0, 1], sin_1) or np.allclose(unitary[0, 1], sin_2)
            bottom_left = np.allclose(unitary[1, 0], unitary[0, 1])
            bottom_right = np.allclose(unitary[1, 1], cos)
            return top_right and bottom_left and bottom_right