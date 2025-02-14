from pytket import Circuit
from bqskit.ir import Circuit as BQCircuit
from pytket.qasm import circuit_from_qasm, circuit_to_qasm
from pytket.passes import FullPeepholeOptimise

from bqskit.ext import pytket_to_bqskit

from pytket.circuit import OpType
from pathlib import Path

import glob
from util import normalized_gp_frob_cost
from collections import Counter
import os
from sys import argv

base_dir = "/pscratch/sd/j/jkalloor/bqskit"

# Run TKET Optimization on initial circuit to get shorter circuits
if __name__ == '__main__':
    input = argv[1] if len(argv) > 1 else "good_blocks"
    input_dir = os.path.join(base_dir, input)
    if not os.path.exists(input_dir):
        print("Invalid input directory", flush=True)
        exit(1)
    output_dir = f"{input_dir}_tket"


    Path(output_dir).mkdir(parents=True, exist_ok=True)

    unoptimized_circ_files = glob.glob(f"{input_dir}/*adder*.qasm")

    for circ_file in unoptimized_circ_files:
        file_name = Path(circ_file).name
        new_circ_file = os.path.join(output_dir, file_name)
        if os.path.exists(new_circ_file):
            continue
        print(circ_file, flush=True)
        circ = circuit_from_qasm(circ_file)
        bqskit_circ = BQCircuit.from_file(circ_file)
        if bqskit_circ.num_qudits <= 12:
            un = bqskit_circ.get_unitary()
            print("Original CX Count: ", circ.n_gates_of_type(OpType.CX), flush=True)
            FullPeepholeOptimise().apply(circ)
            print("Optimized CX Count: ", circ.n_gates_of_type(OpType.CX), flush=True)
            gate_counts = Counter(command.op.type for command in circ.get_commands())
            print("Gate Counts: ", gate_counts, flush=True)
            opt_circ = pytket_to_bqskit(circ)
            print(opt_circ.gate_counts)
            opt_un = opt_circ.get_unitary()
            dist = normalized_gp_frob_cost(un, opt_un)
            print("Normalized Distance: ", normalized_gp_frob_cost(un, opt_un), flush=True)
            if dist < 1e-8:
                print("Writing optimized circuit", flush=True)
                file_name = Path(circ_file).name
                new_circ_file = os.path.join(output_dir, file_name)
                circuit_to_qasm(circ, new_circ_file)
                print(f"Optimized {circ_file}", flush=True)
            else:
                print("Does not meet distance threshold", flush=True)