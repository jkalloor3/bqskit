from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import CircuitGate
import logging
import numpy as np
from sys import argv
import os

from qiskit.quantum_info import SparsePauliOp

import pickle

from bqskit.compiler import Compiler
from bqskit.passes import ScanPartitioner


from common.io import (load_block, get_block_names)
from util import load_circuit, UnitaryDMEvaluator, get_sub_block_nums

def generate_lgt_hamiltonian(num_qubits: int, x: int) -> np.ndarray:
    '''
    H = He (electric) + Hb (magnetic)
    
    He = 3/8 * (3N + 1) - 9/8 * (Z_0 + Z_{N-1}) - 3/4 (sum_{n=1}^{N-2} Z_n)
    - 3/8 * (sum_{n=0}^{N-2} Z_n Z_{n+1})

    Hb = -x/2 (3 + Z_1)(X_0) - x/2 (3 + Z_{N-2})(X_{N-1}) - 
    [x/8 (sum_{n=1}^{N-2} (9 + 3Z_{n-1} + 3Z_{n+1} + Z_{n-1}Z_{n+1}))(X_n))
    '''

    # Generate He
    Z_0_term = ("Z" + "I" * (num_qubits - 1), -9/8)
    Z_N1_term = ("I" * (num_qubits - 1) + "Z", -9/8)
    He = [
        Z_0_term,
        Z_N1_term
    ]

    for i in range(1, num_qubits - 1):
        Z_n_term = ("I" * i + "Z" + "I" * (num_qubits - i - 1), -3/4)
        He.append(Z_n_term)
    
    for i in range(num_qubits - 1):
        Z_nZ_n1_term = ("I" * i + "ZZ" + "I" * (num_qubits - i - 2), -3/8)
        He.append(Z_nZ_n1_term)

    # Generate Hb
    X_0_term = ("X" + "I" * (num_qubits - 1), -x/2 * (3))
    X_0_Z_1_term = ("XZ" + "I" * (num_qubits - 2), -x/2)
    X_N1_term = ("I" * (num_qubits - 1) + "X", -x/2 * (3))
    X_N1_Z_N2_term = ("I" * (num_qubits - 2) + "ZX", -x/2)
    Hb = [
        X_0_term,
        X_0_Z_1_term,
        X_N1_term,
        X_N1_Z_N2_term
    ]

    for i in range(1, num_qubits - 1):
        X_term = ("I" * i + "X" + "I" * (num_qubits - i - 1), -9*x/8)
        # 3Z_{n-1}*X_n
        ZX_term = ("I" * (i - 1) + "ZX" + "I" * (num_qubits - i - 1), -3*x/8)
        # 3Z_{n+1}*X_n
        XZ_term = ("I" * i + "XZ" + "I" * (num_qubits - i - 2), -3*x/8)
        ZXZ_term = ("I" * (i - 1) + "ZXZ" + "I" * (num_qubits - i - 2), -x/8)

        Hb.extend(
            [
                X_term,
                ZX_term,
                XZ_term,
                ZXZ_term
            ]
        )

    op = SparsePauliOp.from_list(He + Hb)
    return op.to_matrix()

def generate_tfim_hamiltonian(num_qubits: int) -> np.ndarray:
    '''
    1D Transverse Field Ising Model Hamiltonian:
    H = J * sum_{i=0}^{N-2} Z_i Z_{i+1} + mu_x * sum_{i=0}^{N-1} X_i
    '''

    Jz = 1.0
    mu_x = 1.0

    He = []
    Hb = []

    for i in range(num_qubits - 1):
        Z_term = ("I" * i + "ZZ" + "I" * (num_qubits - i - 2), Jz)
        He.append(Z_term)

    for i in range(num_qubits):
        X_term = ("I" * i + "X" + "I" * (num_qubits - i - 1), mu_x)
        Hb.append(X_term)

    op = SparsePauliOp.from_list(He + Hb)
    return op.to_matrix()

# Get partitioned circuits for all 8-qubit blocks
def partition_circs(compiler: Compiler,
                    circ_name, 
                    block_num: str) -> tuple[dict[str, Circuit], Circuit]:
    workflow = [
        ScanPartitioner(4)
    ]
    circ_file = load_block(circ_name, block_num, extra="_tket")
    circ = Circuit.from_file(circ_file)
    return compiler.submit(circ, workflow=workflow)


if __name__ == "__main__":
    # circ_names = ["qpe_11", "lgt_11", "qaoa10"]
    # circ_names = ["qpe_11"]
    # circ_names += [f"QITE_8_{i}" for i in range(7)]

    circ_names = ["lgt_11", "mult8"]

    compiler = Compiler(num_workers=128, runtime_log_level=logging.ERROR)

    all_partitioned_ids = {}
    all_partitioned_data = {}

    partitioned_data_file = "partitioned_data_all_circs.pickle"
    missing_circ_names = circ_names.copy()
    if os.path.exists(partitioned_data_file):
        with open(partitioned_data_file, "rb") as f:
            all_partitioned_data = pickle.load(f)
        print("Loaded partitioned data from file.", list(all_partitioned_data.keys()))
        # Only run on circs that are not in data
        missing_circ_names = [name for name in circ_names if name not in all_partitioned_data]

    for circ_name in missing_circ_names:
        all_partitioned_ids[circ_name] = {}
        for large_block_num in get_block_names(circ_name, extra="_tket"):
            id = partition_circs(compiler, circ_name, large_block_num)
            
            all_partitioned_ids[circ_name][large_block_num] = id
        
        all_partitioned_data[circ_name] = {}
        checkpoint_folder_form = f"small_block_checkpoints_final_paper_4_clifft_tket/{circ_name}" + "_{large_block_num}_" + f"*/"
        for large_block_num in get_block_names(circ_name, extra="_tket"):
            out_circ = compiler.result(all_partitioned_ids[circ_name][large_block_num])
            base_dir = checkpoint_folder_form.format(large_block_num=large_block_num)
            sub_block_names = get_sub_block_nums(base_dir)
            assert out_circ.num_operations == len(sub_block_names), f"Number of operations in circuit {circ_name} does not match number of sub-blocks: {out_circ.num_operations} != {len(sub_block_names)}"
            sub_block_circs = {}
            for i, op in enumerate(out_circ.operations()):
                assert isinstance(op.gate, CircuitGate)
                block_name = sub_block_names[i]
                sub_block_circs[block_name] = op.gate._circuit
            all_partitioned_data[circ_name][large_block_num] = (sub_block_circs, out_circ)
    
    with open(partitioned_data_file, "wb") as f:
        pickle.dump(all_partitioned_data, f)

    cliff_t = True

    compiler_ids = []
    for circ_name in circ_names:
        full_circ = load_circuit(circ_name)
        full_circ.remove_all_measurements()
        ham = None
        if circ_name.startswith("lgt_"):
            ham = generate_lgt_hamiltonian(full_circ.num_qudits, 2)
        elif circ_name.startswith("QITE_8_"):
            ham = generate_tfim_hamiltonian(full_circ.num_qudits)
        for tol in [1.0, 2.0, 3.0, 4.0, 5.0]:
            checkpoint_folder_form = f"small_block_checkpoints_final_paper_4_clifft_tket/{circ_name}" + "_{large_block_num}_" + f"{tol}/"
            workflow = [
                UnitaryDMEvaluator(
                    circ_name=circ_name,
                    max_tol=tol,
                    partitioned_data=all_partitioned_data[circ_name],
                    checkpoint_form=checkpoint_folder_form,
                    ham=ham,
                    partitioned_circ_file=f"partitioned_circs/{circ_name}.pickle",
                    save_dir=f"ensemble_dms_{circ_name}/",
                    cliff_t=cliff_t
                )
            ]
            compiler.compile(full_circ, workflow=workflow)
            # id = compiler.submit(full_circ, workflow=workflow)
            # compiler_ids.append(id)

    # for id in compiler_ids:
    #     compiler.result(id)