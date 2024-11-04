"""This module implements the IdentityGate."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from bqskit.ir.gates.constantgate import ConstantGate
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.typing import is_valid_radixes

from bqskit.ir.location import CircuitLocation
from bqskit.qis.unitary.unitary import RealVector

class GlobalPhaseGate(ConstantGate):
    """An Identity (No-OP) Gate with a global phase."""

    def __init__(
        self,
        num_qudits: int = 1,
        radixes: Sequence[int] = [],
        global_phase: np.complex128 = 1
    ) -> None:
        """
        Create an IdentityGate with a global phase

        Args:
            num_qudits (int): The number of qudits this gate acts on.

            radixes (Sequence[int]): The number of orthogonal
                states for each qudit. Defaults to qubits.

        Raises:
            ValueError: If `num_qudits` is nonpositive.
        """
        if num_qudits <= 0:
            raise ValueError('Expected positive integer, got %d' % num_qudits)

        if len(radixes) != 0 and not is_valid_radixes(radixes, num_qudits):
            raise TypeError('Invalid radixes.')

        self._num_qudits = num_qudits
        self._radixes = tuple(radixes or [2] * num_qudits)
        self._dim = int(np.prod(self.radixes))
        self._utry = UnitaryMatrix(UnitaryMatrix.identity(self.dim, self.radixes) * global_phase)
        self.global_phase = global_phase
        self._qasm_name = 'identity%d' % self.num_qudits

    def get_qasm(self, params: RealVector, location: CircuitLocation) -> str:
        """Returns the qasm string for this gate."""
        actual_params = [0,0,0]
        return '{}({}) q[{}];\n'.format(
            "u3",
            ', '.join([str(p) for p in actual_params]),
            '], q['.join([str(q) for q in location]),
        ).replace('()', '')

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, GlobalPhaseGate)
            and self.num_qudits == other.num_qudits
            and self.radixes == other.radixes
            and self.global_phase == other.global_phase
        )

    def __hash__(self) -> int:
        return hash((self.num_qudits, self.radixes, self.global_phase))

    def __str__(self) -> str:
        if self.is_qubit_only():
            return f'GlobalPhaseGate({self.num_qudits}, {self.global_phase})'
        else:
            return f'GlobalPhaseGate({self.num_qudits}, {self.radixes}, {self.global_phase})'
