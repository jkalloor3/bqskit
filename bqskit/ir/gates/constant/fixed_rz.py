import numpy as np

from bqskit.ir.gates.constantgate import ConstantGate
from bqskit.ir.gates.qubitgate import QubitGate
from bqskit.qis import UnitaryMatrix

class FixedRZGate(ConstantGate, QubitGate):
    """
    An RZ Gate fixed to a specific angle.
    """

    _num_qudits = 1
    _qasm_name = 'rz'

    def __init__(self, theta: float) -> None:
        pexp = np.exp(1j * theta / 2)
        nexp = np.exp(-1j * theta / 2)
        self._utry =  UnitaryMatrix(
            [
                [nexp, 0],
                [0, pexp],
            ],
        )
