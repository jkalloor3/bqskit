import os
import sys
from pathlib import Path
import numpy as np
from bqskit.ir import Circuit, CircuitPoint, Operation
from bqskit.ir.gates import *
from bqskit.passes import ZXZXZDecomposition, ToU3Pass
from .distance import normalized_gp_frob_cost
import time

pi_over_4_gates = [np.pi * 0.25 * i for i in range(8)]

pi_over_4_circs = ["I", "T", "S", "ZD", "Z", "ZT", "L", "D"]
t_inds = [1, 3, 5, 7]

def get_rz_circ(angle: float) -> Circuit:
    circ = Circuit(1)
    circ.append_gate(RZGate(angle), (0,))
    return circ

def get_t_count(angles: list[float], precisions: list[int]) -> tuple[np.ndarray[int], np.ndarray[str]]:
    assert len(angles) == len(precisions)
    t_counts = np.zeros(len(angles), dtype=np.int32)
    t_strs = np.array([''] * len(angles), dtype="U256")
    for i, angle in enumerate(angles):
        t_str, t_counts[i] = get_clifft_str(angle, precisions[i])
        t_strs[i] = t_str
    # print("T Strs: ", t_strs)
    return t_counts, t_strs

def get_closest_rz(angle: float, precision: int = 5) -> float:
    circ = get_clifft_gates(angle, precision)
    return RZGate.calc_params(circ.get_unitary())

def get_pos_angle(angle: float) -> float:
    return angle % (2 * np.pi)

def get_clifft_str(angle: float, precision: int = 5) -> tuple[str, int]:
    angle = get_pos_angle(angle)
    angle_str = '\"(' + str(angle) + ')\"'
    # print("Attempting to do: ", angle_str)
    tol = np.power(10.0, -1 * precision)
    is_fixed_angle = [np.allclose(angle, x, atol=tol) for x in pi_over_4_gates]
    # print(is_fixed_angle)
    if np.any(is_fixed_angle):
        ind = is_fixed_angle.index(True)
        circ_str = pi_over_4_circs[ind]
        gates = circ_str
        if ind in t_inds:
            count = 1
        else:
            count = 0
    else:
        command = f'~/Downloads/gridsynth -d {precision} {angle_str}'
        result = os.popen(command).read().strip()
        # Go through gates character by character
        gates = result
        count = gates.count('T')
    return gates, count

def get_clifft_circ(gates: str, simulable: bool = False) -> Circuit:
    circ = Circuit(1)
    for gate in gates:
        if gate == 'I':
            circ.append_gate(IdentityGate(), (0,))
        elif gate == 'Z':
            circ.append_gate(ZGate(), (0,))
        elif gate == 'S':
            circ.append_gate(SGate(), (0,))
        elif gate == 'L':
            circ.append_gate(SdgGate(), (0,))
        elif gate == 'T':
            circ.append_gate(TGate(), (0,))
        elif gate == "H":
            circ.append_gate(HGate(), (0,))
        elif gate == "D":
            circ.append_gate(TdgGate(), (0,))
        elif gate == "X":
            circ.append_gate(XGate(), (0,))

    if simulable:
        un = circ.get_unitary()
        circ = Circuit(1)
        circ.append_gate(U3Gate(), (0,), U3Gate().calc_params(un))
        return circ
    else:
        return circ

def get_clifft_gates(angle: float, precision: int = 5) -> Circuit:
    # Get the ZXZXZ gates
    gates, _ = get_clifft_str(angle, precision)

    return get_clifft_circ(gates)

def convert_to_clifft(circ: Circuit, precision: int = 5, simulable: bool = False) -> Circuit:
    target = circ.get_unitary()
    circuit = circ.copy()
    for cycle, op in circuit.operations_with_cycles():
        if op.num_qudits == 1 and op.num_params > 1:
            # First check if it is just an RZ gate
            if op.params[0] == 0 and op.params[1] == 0:
                zxzxz_circ = get_clifft_gates(op.params[2], precision)
            else:
                zxzxz_circ = ZXZXZDecomposition.run_zxzxz_decomp_circ(op.get_unitary())
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

    final_unitary = circuit.get_unitary()

    dist = normalized_gp_frob_cost(target, final_unitary)
    print("Distance after conversion: ", dist, flush=True)

    return circuit

def convert_to_clifft_tbudget(circ: Circuit, t_budget: int, max_precision: int = 6, simulable: bool = False) -> Circuit:
    start_time = time.process_time()
    target = circ.get_unitary()
    circuit = circ.copy()
    zxzxz_circ = ZXZXZDecomposition.get_zxzxz_circ_structure()
    pts_to_replace = []
    new_ops = []
    for cycle, op in circuit.operations_with_cycles():
        if op.num_qudits == 1 and op.num_params > 1:
            # First check if it is just an RZ gate
            pt = CircuitPoint(cycle, op.location[0])
            if op.params[0] == 0 and op.params[1] == 0:
                cliff_str, count = get_clifft_str(op.params[2], max_precision)
                if count > 1:
                    new_op = Operation(RZGate(op.params[2]), op.location, [op.params[2]])
                else:
                    clifft_circ = get_clifft_circ(cliff_str)
                    new_op = Operation(CircuitGate(clifft_circ), op.location, [])
                    new_un = new_op.get_unitary()
                    dist = normalized_gp_frob_cost(op.get_unitary(), new_un)
            else:
                new_params = ZXZXZDecomposition.get_zxzxz_decomp_params(op.get_unitary())
                new_op = Operation(CircuitGate(zxzxz_circ), op.location, new_params)
            pts_to_replace.append(pt)
            new_ops.append(new_op)
    
    circuit.batch_replace(pts_to_replace, new_ops)
    circuit.unfold_all()

    # final_unitary = circuit.get_unitary()

    # dist = normalized_gp_frob_cost(target, final_unitary)
    # print("Distance after 1st conversion: ", dist, flush=True)


    t_budget -= np.sum(circuit.count(TGate()) + circuit.count(TdgGate()))

    # num_params = circuit.num_params
    params = []
    for op in circuit:
        params.extend(op.params)
    num_params = len(params)
    # print("Number of Parameters: ", num_params, flush=True)
    # print(circuit.gate_counts)
    # Calculate the initial precision for the conversion
    min_prec = max(t_budget // num_params // 10, 1)
    precisions = np.array([min_prec] * num_params)
    t_counts, t_strs = get_t_count(params, precisions)
    ts_used = np.sum(t_counts)
    if ts_used > t_budget:
        print(f"Initial t count: {ts_used} exceeds budget: {t_budget}", flush=True)
        return circuit
    remaining_ts = t_budget - ts_used
    # Randomly choose num_params_inc inds to increase precision
    while remaining_ts > 10:
        num_params_inc = min(remaining_ts // 10, num_params)
        inds = np.random.choice(num_params, num_params_inc, replace=False)
        # print(f"Changing {num_params_inc} precisions", flush=True)
        precisions[inds] += 1
        new_precisions = precisions[inds]
        new_params = circuit.params[inds]
        new_t_counts, new_t_strs = get_t_count(new_params, new_precisions)
        t_counts[inds] = new_t_counts
        t_strs[inds] = new_t_strs
        ts_used = np.sum(t_counts)
        # print("New T Count: ", ts_used, flush=True)
        remaining_ts = t_budget - ts_used
    
    # print("Final T Count: ", ts_used + np.sum(circuit.count(TGate()) + circuit.count(TdgGate())), flush=True)

    ind = 0
    pts_to_replace = []
    new_ops = []
    for cycle, op in circuit.operations_with_cycles():
        if isinstance(op.gate, RZGate):
            cliff_str = t_strs[ind]
            clifft_circ = get_clifft_circ(cliff_str, simulable=simulable)
            ind += 1
            pt = CircuitPoint(cycle, op.location[0])
            new_op = Operation(CircuitGate(clifft_circ), op.location, [])
            pts_to_replace.append(pt)
            new_ops.append(new_op)

    circuit.batch_replace(pts_to_replace, new_ops)
    circuit.unfold_all()

    # print(circuit.gate_counts)

    final_unitary = circuit.get_unitary()

    dist = normalized_gp_frob_cost(target, final_unitary)
    final_time = time.process_time() - start_time
    print(f"Time: {final_time} Distance: {dist}", flush=True)
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