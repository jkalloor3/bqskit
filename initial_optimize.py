from pytket import Circuit
from pytket.qasm import circuit_from_qasm, circuit_to_qasm
from pytket.passes import FullPeepholeOptimise

from bqskit.ext import pytket_to_bqskit

from pytket.circuit import OpType
from pathlib import Path

import glob
from bqskit.ir.gates import GlobalPhaseGate



# Run TKET Optimization on initial circuit to get shorter circuits
if __name__ == '__main__':
    
    unoptimized_circ_files = glob.glob("/pscratch/sd/j/jkalloor/bqskit/qce23_qfactor_benchmarks/adder9.qasm")

    for circ_file in unoptimized_circ_files:
        circ = circuit_from_qasm(circ_file)
        print("Original CX Count: ", circ.n_gates_of_type(OpType.CX))
        if circ.n_qubits <= 10:
            un = pytket_to_bqskit(circ).get_unitary()
        FullPeepholeOptimise().apply(circ)
        print("Optimized CX Count: ", circ.n_gates_of_type(OpType.CX))

        if circ.n_qubits <= 10:
            opt_circ = pytket_to_bqskit(circ)
            opt_un = opt_circ.get_unitary()
            global_phase_correction = un.get_target_correction_factor(opt_un)
            opt_circ.append_gate(GlobalPhaseGate(1, global_phase=global_phase_correction), (0,))
            opt_un = opt_circ.get_unitary()
            print("Frobenius Distance: ", un.get_frobenius_distance(opt_un))
            print("Normalized Distance: ", un.get_distance_from(opt_un))

        
        circ_file = circ_file.replace("qce23_qfactor_benchmarks", "ensemble_benchmarks_opt")
        Path(circ_file).parent.mkdir(parents=True, exist_ok=True)
        circuit_to_qasm(circ, circ_file)
        print(f"Optimized {circ_file}")