from pytket import Circuit
from bqskit.ir import Circuit as BQCircuit
from pytket.qasm import circuit_from_qasm, circuit_to_qasm, circuit_to_qasm_str
from bqskit.ir.lang.qasm2 import OPENQASM2Language
from pytket.passes import FullPeepholeOptimise, SynthesiseTK, KAKDecomposition

from bqskit.ext import pytket_to_bqskit

from pytket.circuit import OpType
from pathlib import Path

import glob
from util import normalized_gp_frob_cost
from collections import Counter
import os
import time
from sys import argv

base_dir = "/pscratch/sd/j/jkalloor/ensemble_paper/bqskit"

temp_file = "temp.qasm"

lang = OPENQASM2Language()

def full_optimization(circ_file: str, new_circ_file: str) -> bool:
    '''
    Try to do a full optimization on the circuit, if the final 
    unitary distance is not far, then write the optimized circuit
    '''
    print(f"Optimizing {circ_file}", flush=True)
    bqskit_circ = BQCircuit.from_file(circ_file)
    try:
        circ = circuit_from_qasm(circ_file, maxwidth=bqskit_circ.num_qudits)
    except Exception as e:
        print(f"Error reading circuit from {circ_file}: {e}", flush=True)
        return True
    print("Original Gate Counts: ", bqskit_circ.gate_counts, flush=True)
    if bqskit_circ.num_operations > 20000:
        print(f"{circ_file} has too many operations, skipping", flush=True)
        return True
    # if bqskit_circ.num_qudits > 12:
    #     print(f"{circ_file} has too many qudits, skipping", flush=True)
    #     return False
    # un = bqskit_circ.get_unitary()
    # Cannot swap for now as blocks must connect as normal -> Further Optimization
    FullPeepholeOptimise(allow_swaps=False).apply(circ)
    print("Optimized CX Count: ", circ.n_gates_of_type(OpType.CX), flush=True)
    gate_counts = Counter(command.op.type for command in circ.get_commands())
    print("Gate Counts: ", gate_counts, flush=True)
    circuit_to_qasm(circ, temp_file, maxwidth=bqskit_circ.num_qudits)
    # opt_circ = lang.decode(circuit_to_qasm_str(circ))
    time.sleep(2)
    # opt_circ = BQCircuit.from_file(temp_file)
    opt_tket_circ = circuit_from_qasm(temp_file, maxwidth=bqskit_circ.num_qudits)
    # print(opt_circ.gate_counts)
    # opt_un = opt_circ.get_unitary()
    print(opt_tket_circ.n_gates)
    # opt_tket_un = opt_tket_circ.get_unitary()
    # dist = normalized_gp_frob_cost(opt_un, un)
    # dist_2 = normalized_gp_frob_cost(opt_tket_un, un)
    # print("Normalized Distance: ", dist, flush=True)
    # print("TKET Unitary Distance: ", dist_2, flush=True)
    # if dist < 1e-8:
    file_name = Path(circ_file).name
    new_circ_file = os.path.join(output_dir, file_name)
    print("Writing to ", new_circ_file, flush=True)
    circuit_to_qasm(circ, new_circ_file, maxwidth=bqskit_circ.num_qudits)
    print(f"Optimized {circ_file}", flush=True)
    return True
    # else:
    #     return False

def faster_opt(circ_file: str, new_circ_file: str) -> bool:
    '''
    Try to do a faster optimization on the circuit, if the final 
    unitary distance is not far, then write the optimized circuit
    '''
    circ = circuit_from_qasm(circ_file)
    print("Original CX Count: ", circ.n_gates_of_type(OpType.CX), flush=True)
    SynthesiseTK().apply(circ)
    KAKDecomposition().apply(circ)
    print("Optimized CX Count: ", circ.n_gates_of_type(OpType.CX), flush=True)
    if circ.n_qubits < 12:
        target_circ = BQCircuit.from_file(circ_file)
        target = target_circ.get_unitary()
        opt_circ = pytket_to_bqskit(circ)
        # print(opt_circ.gate_counts)
        opt_un = opt_circ.get_unitary()
        dist = normalized_gp_frob_cost(target, opt_un)
        print("Normalized Distance: ", normalized_gp_frob_cost(target, opt_un), flush=True)
        if dist > 1e-8:
            return False
        else:
            circuit_to_qasm(circ, new_circ_file)
            print(f"Quick Optimized {circ_file}", flush=True)
            return True

# Run TKET Optimization on initial circuit to get shorter circuits
if __name__ == '__main__':
    input = argv[1] if len(argv) > 1 else "good_blocks"
    clifft = bool(argv[2]) if len(argv) > 2 else False
    input_dir = os.path.join(base_dir, input)
    if not os.path.exists(input_dir):
        print("Invalid input directory", flush=True)
        exit(1)
    output_dir = f"{input_dir}_tket"


    Path(output_dir).mkdir(parents=True, exist_ok=True)

    circ_types = ["*lgt_380*"]
    unoptimized_circ_files = []
    for circ_type in circ_types:
        unoptimized_circ_files.extend(glob.glob(f"{input_dir}/{circ_type}*.qasm"))

    print(unoptimized_circ_files, flush=True)

    for circ_file in unoptimized_circ_files:
        file_name = Path(circ_file).name
        new_circ_file = os.path.join(output_dir, file_name)
        if os.path.exists(new_circ_file):
            # print file and read CNOT counts
            cnot_count = circuit_from_qasm(new_circ_file).n_gates_of_type(OpType.CX)
            print(f"Skipping {circ_file}, already optimized with {cnot_count} CNOTs", flush=True)
            continue
        full_opt = full_optimization(circ_file, new_circ_file)
        if not full_opt:
            print(f"Failed to optimize {circ_file}", flush=True)
            # Try a faster optimization 
            faster_opt(circ_file, new_circ_file)
        