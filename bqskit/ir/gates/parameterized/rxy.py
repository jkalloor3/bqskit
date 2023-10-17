"""This module implements the RYYGate."""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

from bqskit.ir.gates.qubitgate import QubitGate
from bqskit.qis.unitary.differentiable import DifferentiableUnitary
from bqskit.qis.unitary.unitary import RealVector
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.cachedclass import CachedClass


class RXYGate(
    QubitGate,
    DifferentiableUnitary,
    CachedClass,
):
    """
    A gate representing an arbitrary rotation around the XY axis as defined in:

    https://pyquil-docs.rigetti.com/en/v2.28.0/apidocs/autogen/pyquil.gates.XY.html
    """

    _num_qudits = 2
    _num_params = 1
    _qasm_name = 'rxy'

    def get_unitary(self, params: RealVector = []) -> UnitaryMatrix:
        """Return the unitary for this gate, see :class:`Unitary` for more."""
        self.check_parameters(params)

        cos = np.cos(params[0] / 2)
        psin = 1j * np.sin(params[0] / 2)

        return UnitaryMatrix(
            [
                [1, 0, 0, 0],
                [0, cos, psin, 0],
                [0, psin, cos, 0],
                [0, 0, 0, 1],
            ],
        )

    def get_grad(self, params: RealVector = []) -> npt.NDArray[np.complex128]:
        """
        Return the gradient for this gate.

        See :class:`DifferentiableUnitary` for more info.
        """
        self.check_parameters(params)

        dcos = -np.sin(params[0] / 2) / 2

        dpsin = 1j * np.cos(params[0] / 2) / 2

        return np.array(
            [
                [
                    [1, 0, 0, 0],
                    [0, dcos, dpsin, 0],
                    [0, dpsin, dcos, 0],
                    [0, 0, 0, 1],
                ],
            ], dtype=np.complex128,
        )
