from util.counter import load_avg_ensemble_counts_est, load_avg_ensemble_counts_full, get_circ_counts
from bqskit.ir import Circuit
import numpy as np
from bqskit.ir.lang.qasm2 import OPENQASM2Language
from util.gg import gg_gate_def, GridSynthGate
import time

qlang = OPENQASM2Language(gate_defs=[("gg", gg_gate_def)])

if __name__ == '__main__':
    
    circ_file = "good_blocks_tket/qae11_1.qasm"
    circ = Circuit.from_file(circ_file)

    print(circ.gate_counts)

    circ_files = ['good_blocks_tket/qae11_1.qasm', 'good_blocks_tket/qae11_3.qasm', 'bad_blocks_tket/qae11_0.qasm', 'bad_blocks_tket/qae11_2.qasm']

    # circ_files = [circ_file]

    orig_t = get_circ_counts(circ_files[:1], count_t=True, target_error=1e-6)
    orig_cx = get_circ_counts(circ_files[:1], target_error=1e-6)

    print("Original CX: ", orig_cx)
    print("Original T: ", orig_t)
    exit(0)


    # Now test ensemble counts
    ensemble_file = "block_checkpoints_final_paper_clifft_tket/qae11_1_3.0/ensemble_0_.qasms"
    jiggle_file = "block_checkpoints_final_paper_clifft_tket/qae11_1_3.0/ensemble_0_jiggles_.npy"

    start = time.time()
    est_ens_t = load_avg_ensemble_counts_est(ensemble_file, jiggle_file, 1e-6, count_t=True)
    est_time = time.time() - start
    print("Est Time: ", est_time)
    start = time.time()
    full_ens_t = load_avg_ensemble_counts_full(ensemble_file, jiggle_file, 1e-6, count_t=True)
    full_time = time.time() - start
    print("Full Time: ", full_time)

    orig_t_2 = load_avg_ensemble_counts_full(circ_file, None, 1e-6, count_t=True)

    print("Original T: ", orig_t)
    print("Original T 2: ", orig_t_2)
    print("Est Ensemble T: ", est_ens_t)
    print("Full Ensemble T: ", full_ens_t)

    est_ens_cx = load_avg_ensemble_counts_est(ensemble_file, None, 1e-6)
    full_ens_cx = load_avg_ensemble_counts_full(ensemble_file, None, 1e-6)

    # print some random counts
    with open(ensemble_file, "r") as f:
        qasms = f.read().split("\nBREAK\n")

    random_circs = qasms[:3]

    params = np.load(jiggle_file)

    all_circs = []

    for i, qasm in enumerate(random_circs):
        circ = qlang.decode(qasm)
        circ.set_params(params[i][0])
        print("Circ Gate Counts: ", circ.gate_counts)
        gg_precs = []
        for op in circ.operations():
            if isinstance(op.gate, GridSynthGate):
                gg_precs.append(op.params[1])
        print("Avg GG Prec: ", np.mean(gg_precs))
        all_circs.append(circ)

    # print("Original CX: ", orig_cx)
    # print("Est Ensemble CX: ", est_ens_cx)
    # print("Full Ensemble CX: ", full_ens_cx)

    

