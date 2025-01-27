import os
import sys
from pathlib import Path
import numpy as np
from bqskit.ir import Circuit, CircuitPoint
from bqskit.ir.gates import *
from bqskit.passes import ZXZXZDecomposition, ToU3Pass

pi_over_4_gates = [np.pi * 0.25 * i for i in range(8)]

pi_over_4_circs = ["I", "T", "S", "Z Tdg", "Z", "Z T", "Sdg", "Tdg"]


def get_closest_rz(angle: float, precision: int = 5) -> float:
    circ = get_clifft_gates(angle, precision)
    return RZGate.calc_params(circ.get_unitary())


def get_pos_angle(angle: float) -> float:
    return angle % (2 * np.pi)

def get_clifft_gates(angle: float, precision: int = 5) -> Circuit:
    # Get the ZXZXZ gates
    circ = Circuit(1)
    angle = get_pos_angle(angle)
    angle_str = '\"(' + str(angle) + ')\"'
    # print("Attempting to do: ", angle_str)
    is_fixed_angle = [np.allclose(angle, x, atol=10**(-1 * precision)) for x in pi_over_4_gates]
    # print(is_fixed_angle)
    if np.allclose(angle, 0, atol=10**(-1 * precision)):
        return circ
    elif np.allclose(angle, np.pi, atol=10**(-1 * precision)):
        circ.append_gate(ZGate(), (0,))
        return circ
    elif np.any(is_fixed_angle):
        ind = is_fixed_angle.index(True)
        circ_str = pi_over_4_circs[ind]
        gates = circ_str.split()
        # print(gates)
    else:
        command = f'~/Downloads/gridsynth -d {precision} {angle_str}'
        result = os.popen(command).read().strip()
        # Go through gates character by character
        gates = result

    for gate in gates:
        if gate == 'I':
            circ.append_gate(IdentityGate(), (0,))
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


def convert_to_clifft(circuit: Circuit, precision: int = 5, simulable: bool = False) -> Circuit:
    for cycle, op in circuit.operations_with_cycles():
        if op.num_qudits == 1 and op.num_params > 1:
            # First check if it is just an RZ gate
            if op.params[0] == 0 and op.params[1] == 0:
                zxzxz_circ = get_clifft_gates(op.params[2], precision)
            else:
                zxzxz_circ = ZXZXZDecomposition.run_zxzxz_decomp_circ(op.get_unitary(), )
            pt = CircuitPoint(cycle, op.location[0])
            circuit.replace_with_circuit(pt, zxzxz_circ,as_circuit_gate=True)

    circuit.unfold_all()

    for cycle, op in circuit.operations_with_cycles():
        if isinstance(op.gate, RZGate):
            angle = op.params[0]
            if simulable:
                new_angle = get_closest_rz(angle, precision)
                op.params[0] = new_angle
            else:
                clifft_circ = get_clifft_gates(angle, precision)
                pt = CircuitPoint(cycle, op.location[0])
                circuit.replace_with_circuit(pt, clifft_circ, as_circuit_gate=True)

    circuit.unfold_all()

    if simulable:
        ToU3Pass.run_group_circ(circuit)

    return circuit
    


def read_qasm_files(folder_path) -> list[tuple[str, Circuit]]:
    circuits = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.qasm'):
            circ_name = filename.split('.')[0]
            file_path = os.path.join(folder_path, filename)
            circuit = Circuit.from_file(file_path)
            circuits.append((circ_name, circuit))
    return circuits


if __name__ == '__main__':
    # Example usage
    precision = int(sys.argv[1])
    simulable = bool(int(sys.argv[2])) if len(sys.argv) > 2 else False
    sim_str = "_sim" if simulable else ""
    folder_path = "/Users/jkalloor3/BQSKit/bqskit/ensemble_benchmarks"
    output_folder = f"/Users/jkalloor3/BQSKit/bqskit/clifft_benchmarks_{precision}{sim_str}"
    circuits = read_qasm_files(folder_path)
    for name, circuit in circuits:
        cliff_circuit = convert_to_clifft(circuit, precision, simulable=simulable)
        output_file = os.path.join(output_folder, name + '.qasm')
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        cliff_circuit.save(output_file)