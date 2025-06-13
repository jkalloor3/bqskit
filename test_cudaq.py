from sim_lib.sim_lib import (generate_lgt_hamiltonian_cudaq, cuda_kernel,
                             create_ham_args_cudaq, run_cudaq_nisq_circs)
from common.io import load_block
from bqskit.ir import Circuit
import numpy as np
import cudaq
from mpi4py import MPI

if __name__ == '__main__':
    circ_name = "lgt_11"
    large_block_num = "0"
    initial_circ_file = load_block(circ_name, large_block_num, extra="_tket")
    qasm = open(initial_circ_file, 'r').read()
    initial_circ: Circuit = Circuit.from_file(initial_circ_file)
    print("Original CX Count: ", qasm.count("cx "), flush=True)

    mpi_rank = MPI.COMM_WORLD.Get_rank()
    mpi_size = MPI.COMM_WORLD.Get_size()

    ham = generate_lgt_hamiltonian_cudaq(11, 2)
    cudaq.set_target('nvidia')
    expectations = []
    circs = [initial_circ.copy() for _ in range(5)]
    avg_exp = run_cudaq_nisq_circs(circs, ham, add_coherent_error=1e-6,
                                   num_shots=4096, use_noise=True, average=True)

    print(mpi_rank, ":", avg_exp)