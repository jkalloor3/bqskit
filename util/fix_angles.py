from bqskit.compiler.passdata import PassData
from bqskit.compiler.basepass import BasePass
from bqskit.ir import Circuit, CircuitPoint, Operation
from bqskit.ir.gates import *
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
    def __init__(self, precision=5, run_scan_sols: bool = False):
        self.precision = precision
        self.run_scan_sols = run_scan_sols


    def run_circ(circuit: Circuit, precision: int) -> None:

        if circuit.num_params == 0:
            return # No params to try and fix

        precision = precision + np.log10(circuit.num_params)
        for cycle, op in circuit.operations_with_cycles():
            if op.num_qudits == 1:
                if isinstance(op.gate, RXGate):
                    if not RXGate.is_rx(op.get_unitary()):
                        RXGate.is_rx(op.get_unitary(), verbose=True)
                        print("RX Gate not RX")
                        print(op.get_unitary())
                        exit(1)
                if isinstance(op.gate, RYGate):
                    if not RYGate.is_ry(op.get_unitary()):
                        RYGate.is_ry(op.get_unitary(), verbose=True)
                        print("RY Gate not RY")
                        print(op.get_unitary())
                        exit(1)

                if RZGate.is_rz(op.get_unitary()):
                    angle = RZGate.calc_params(op.get_unitary())
                    # Either fix gate or replace with RZ gate
                    zxzxz_circ = get_rz_gate_circ(angle, precision)
                    pt = CircuitPoint(cycle, op.location[0])
                    circuit.replace_with_circuit(pt, zxzxz_circ,as_circuit_gate=True)
                    # new_dist = normalized_gp_frob_cost(zxzxz_circ.get_unitary(), op.get_unitary())
                elif RXGate.is_rx(op.get_unitary()):
                    rz_unitary = HGate().get_unitary() @ op.get_unitary() @ HGate().get_unitary()
                    angle = RZGate.calc_params(rz_unitary)
                    zxzxz_circ = get_rz_gate_circ(angle, precision)
                    # Add hadamards on both sides
                    zxzxz_circ.insert_gate(0, HGate(), (0,))
                    zxzxz_circ.append_gate(HGate(), (0,))
                    pt = CircuitPoint(cycle, op.location[0])
                    circuit.replace_with_circuit(pt, zxzxz_circ,as_circuit_gate=True)
                elif RYGate.is_ry(op.get_unitary()):
                    hy_unitary = SGate().get_unitary() @ HGate().get_unitary()
                    rz_unitary = hy_unitary.conj().T @ op.get_unitary() @ hy_unitary
                    angle = RZGate.calc_params(rz_unitary)
                    zxzxz_circ = get_rz_gate_circ(angle, precision)
                    # Add SH gate on both sides
                    zxzxz_circ.insert_gate(0, HGate(), (0,))
                    zxzxz_circ.insert_gate(0, SdgGate(), (0,))
                    zxzxz_circ.append_gate(HGate(), (0,))
                    zxzxz_circ.append_gate(SGate(), (0,))
                    pt = CircuitPoint(cycle, op.location[0])
                    circuit.replace_with_circuit(pt, zxzxz_circ,as_circuit_gate=True)
                    # new_dist = normalized_gp_frob_cost(zxzxz_circ.get_unitary(), op.get_unitary())

        circuit.unfold_all()

    async def run(self, circuit: Circuit, data: PassData) -> None:
        if self.run_scan_sols:
            init_dists = [x[1] for x in data["scan_sols"]]
            orig_gate_counts = [x[0].gate_counts for x in data["scan_sols"]]
            targets = [x[0].get_unitary() for x in data["scan_sols"]]
            for circ, _ in data["scan_sols"]:
                FixAnglesPass.run_circ(circ, self.precision)
            final_dists = np.array([normalized_gp_frob_cost(c[0].get_unitary(), target) for target, c in zip(targets, data["scan_sols"])])
            if np.any(final_dists > 0.001):
                print("Circ Gates: ", orig_gate_counts, 
                      [x[0].gate_counts for x in data["scan_sols"]], flush=True)
                print("Orig Gate Counts: ", circuit.gate_counts, flush=True)
                print("Post-fix Angles: ", init_dists, final_dists, flush=True)
        else:
            FixAnglesPass.run_circ(circuit, self.precision)

class UnFixTPass(BasePass):

    async def run(self, circuit: Circuit, data: PassData) -> None:
        pts = []
        new_ops = []
        for cycle, op in circuit.operations_with_cycles():
            if op.num_qudits == 1:
                pt = CircuitPoint(cycle, op.location[0])
                if isinstance(op.gate, TGate) or isinstance(op.gate, TdgGate):
                    if isinstance(op.gate, TdgGate):
                        angle = -np.pi / 4
                    else:
                        angle = np.pi / 4
                    gate = RZGate()
                    new_ops.append(Operation(gate, op.location, [angle]))
                    pts.append(pt)
        
        circuit.batch_replace(pts, new_ops)
        circuit.unfold_all()
        # print("Unfixing Params: ", circuit.num_params, flush=True)
        # print("Distance from target: ", normalized_gp_frob_cost(circuit.get_unitary(), data.target))
        return