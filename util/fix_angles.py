from bqskit.compiler.passdata import PassData
from bqskit.compiler.basepass import BasePass
from bqskit.ir import Circuit, CircuitPoint
from bqskit.ir.gates import *
from bqskit.passes import ZXZXZDecomposition, ToU3Pass
import numpy as np

from .distance import normalized_gp_frob_cost


PI_over_4_gates = [np.pi * 0.25 * i for i in range(8)]

PI_over_4_circs = ["I", "T", "S", "Z Tdg", "Z", "Z T", "Sdg", "Tdg"]

def get_rz_gate_circ(angle, precision):
    circ = Circuit(1)
    mod_angle = angle % (2 * np.pi)
    is_fixed_angle = [np.allclose(mod_angle, x, 
                                  atol=10**(-1 * precision)) for x in PI_over_4_gates]
    
    if np.any(is_fixed_angle):
        ind = is_fixed_angle.index(True)
        circ_str = PI_over_4_circs[ind]
        gates = circ_str.split()
    else:
        # Not a fixed angle
        gate = RZGate()
        circ.append_gate(gate, (0,), [angle])
        return circ
    
    for gate in gates:
        if gate == 'I':
            continue
        elif gate == 'Z':
            circ.append_gate(ZGate(), (0,))
        elif gate == 'S':
            circ.append_gate(SGate(), (0,))
        elif gate == 'Sdg':
            circ.append_gate(SdgGate(), (0,))
        elif gate == 'T':
            circ.append_gate(TGate(), (0,))
        elif gate == "H":
            circ.append_gate(HGate(), (0,))
        elif gate == "Tdg":
            circ.append_gate(TdgGate(), (0,))
        elif gate == "X":
            circ.append_gate(XGate(), (0,))
    return circ

class FixAnglesPass(BasePass):
    def __init__(self, precision=5):
        self.precision = precision


    async def run(self, circuit: Circuit, data: PassData) -> None:
        for cycle, op in circuit.operations_with_cycles():
            if op.num_qudits == 1 and RZGate.is_rz(op.get_unitary()):
                angle = RZGate.calc_params(op.get_unitary())
                # Either fix gate or replace with RZ gate
                zxzxz_circ = get_rz_gate_circ(angle, self.precision)
                pt = CircuitPoint(cycle, op.location[0])
                circuit.replace_with_circuit(pt, zxzxz_circ,as_circuit_gate=True)

        circuit.unfold_all()

        print("Num Params after fixing angles: ", circuit.num_params)
        print("Distance from target: ", normalized_gp_frob_cost(circuit.get_unitary(), data.target))
        return circuit