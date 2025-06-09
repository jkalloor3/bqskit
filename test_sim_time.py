from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import CNOTGate, CircuitGate
from sys import argv
import glob
import numpy as np
import os

from concurrent.futures import ProcessPoolExecutor, as_completed

from pathlib import Path
from multiprocessing.shared_memory import SharedMemory

from bqskit.compiler import Compiler
from bqskit.passes import ScanPartitioner
from bqskit.ir.point import CircuitPoint

from bqskit.ir.lang.qasm2 import OPENQASM2Language

import csv
from util import (load_block, load_ensemble, load_avg_ensemble_counts_full,
                  get_block_names)
from util.fix_global_phase import fix_phase

from qiskit import QuantumCircuit

from qiskit.quantum_info import SparsePauliOp
import time

from sim_lib.sim_lib import (run_nisq_circs, create_ham_args, 
                          run_single_nisq_circ)

shots = 1048576  
circ_name = "lgt_11"
checkpoint_folder = f"small_block_checkpoints_tket/{circ_name}"

lang = OPENQASM2Language()


USE_QP = True


# Get partitioned circuits for all 8-qubit blocks
def partition_circs(compiler: Compiler, block_num: str, 
                     sub_block_names: list[str]) -> tuple[dict[str, Circuit], Circuit]:
    workflow = [
        ScanPartitioner(4)
    ]
    circ_file = load_block(circ_name, block_num, extra="_tket")
    circ = Circuit.from_file(circ_file)
    out_circ = compiler.compile(circ, workflow=workflow)
    assert out_circ.num_operations == len(sub_block_names)
    sub_block_circs = {}
    for i, op in enumerate(out_circ.operations()):
        assert isinstance(op.gate, CircuitGate)
        block_name = sub_block_names[i]
        sub_block_circs[block_name] = op.gate._circuit

    return sub_block_circs, out_circ
        
def get_sub_block_nums(block_num: str) -> list[str]:
    sub_block_path = f"{checkpoint_folder}_{block_num}_*/block_*.data"
    sub_block_files = glob.glob(sub_block_path)
    sub_block_nums = set()
    for sub_block_file in sub_block_files:
        sub_block_num = Path(sub_block_file).name.split("_")[-1].split(".")[0]
        sub_block_nums.add(sub_block_num)
    sub_block_nums = sorted(list(sub_block_nums))
    return sub_block_nums

def get_sub_block_count(large_block_num: str, small_block_num: str, max_tol: float) -> tuple[bool, int]:
    qasm_file = f"{checkpoint_folder}_{large_block_num}_{max_tol}/block_{small_block_num}/ensemble_final.qasms"
    csv_file = f"{checkpoint_folder}_{large_block_num}_{max_tol}/block_{small_block_num}.csv"
    # Read CSV file, if the ratio is < 20 then we can read counts
    if not os.path.exists(csv_file):
        return False, 0
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        min_ratio = float("inf")
        final_frob_cost = float("inf")
        for row in reader:
            if "Ratio" in row:  # Check if the column value is not empty
                min_ratio = min(min_ratio, float(row["Ratio"]))
                final_frob_cost = min(final_frob_cost, float(row["Norm. Bias"]))
        
    # Now we need to check if the ratio is less than 20
    if min_ratio > 5:
        return False, 0
    # Now we need to get the avg. number of CNOTs
    count = load_avg_ensemble_counts_full(qasm_file,
                                          None,
                                          None,
                                          target_error=max_tol,
                                          count_t=False,
                                          count_rz=False)
    return True, count

def get_random_circs(num_circs: int,
                    block_circs: list[Circuit],
                    probs: np.ndarray, 
                    shm: SharedMemory,
                    param_shape: tuple[int]) -> list[Circuit]:
    '''
    Generate a random circuit from the block_circs.
    '''
    assert len(block_circs) > 1
    # Now pick a random param from the shared memory
    shm_array = np.ndarray(param_shape, dtype=np.float64, buffer=shm.buf)

    # Else, we are using an ensemble
    if USE_QP:
        rand_inds = np.random.choice(shm_array.shape[0] * shm_array.shape[1], size=num_circs, p=probs)
        num_params_per_circ = shm_array.shape[1]
        rand_circ_inds = [i // num_params_per_circ for i in rand_inds]
        rand_param_inds = [i % num_params_per_circ for i in rand_inds]
    else:
        rand_circ_inds = np.random.randint(0, len(block_circs), size=num_circs)
        rand_param_inds = np.random.randint(0, shm_array.shape[1], size=num_circs)
    circ_strs = []
    # uns = []
    for c_ind, p_ind in zip(rand_circ_inds, rand_param_inds):
        # Now we need to get the random circuits
        rand_circ: Circuit = block_circs[c_ind].copy()
        circ_param = shm_array[c_ind][p_ind]
        # Now we need to set the params of the circuit
        rand_circ.set_params(circ_param)
        # fix_phase(rand_circ, target)
        # uns.append(rand_circ.get_unitary())
        circ_strs.append(rand_circ)

    return circ_strs

def generate_hamiltonian(num_qubits: int, x: int) -> SparsePauliOp:
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
    return op

ham = generate_hamiltonian(11, 2)


def generate_small_block_circuits(block_circs: dict[str, list[Circuit]], 
                           block_names: list,
                           block_probs: dict[str, np.ndarray],
                           param_shapes: dict[str, tuple[int]],
                           shms: dict[str, SharedMemory],
                           pcirc: Circuit,
                           ens_size: int) -> list[Circuit]:
    all_circs = []

    circ_blocks = {}

    for block_name in shms.keys():
        shm = shms.get(block_name, None)
        param_shape = param_shapes.get(block_name, None)
        probs = block_probs.get(block_name, None)
        circ_blocks[block_name] = get_random_circs(ens_size,
                                                block_circs[block_name],
                                                probs=probs, 
                                                shm=shm,
                                                param_shape=param_shape)
        
    target = pcirc.get_unitary()

    for i in range(ens_size):
        circ = pcirc.copy()
        ind = 0
        for cycle, op in circ.operations_with_cycles():
            pt = CircuitPoint(cycle, op.location[0])
            block_name = block_names[ind]
            ind += 1
            assert isinstance(op.gate, CircuitGate)
            assert isinstance(op.gate._circuit, Circuit)
            if block_name in circ_blocks:
                new_block_circ = circ_blocks[block_name][i]
                assert isinstance(new_block_circ, Circuit)
                assert op.gate._circuit.num_qudits == new_block_circ.num_qudits
                circ.replace_with_circuit(pt, new_block_circ, 
                                          as_circuit_gate=True)
        fix_phase(circ, target)
        all_circs.append(circ)
    # print("Finish Generating Circs: ", time.time() - start, flush=True)
    return all_circs

def create_shared_memory(large_block_num: str, 
                         small_block_num: str, 
                         max_tol: float) -> tuple[SharedMemory, tuple[int]]:
    # Get params_file
    params_file = f"{checkpoint_folder}_{large_block_num}_{max_tol}/block_{small_block_num}/ensemble_final_jiggle.npy"
    params: np.ndarray = np.load(params_file)
    param_shape = params.shape
    shm_name = f"{circ_name}_{large_block_num}_{small_block_num}_{max_tol}"
    float_size = np.dtype(np.float64).itemsize
    try:
        shm = SharedMemory(name=shm_name, 
                        create=True, 
                        size=np.prod(param_shape)*float_size)
    except:
        print("Shared Memory already exists: ", shm_name)
        shm = SharedMemory(name=shm_name, create=False)
        shm.close()
        shm.unlink()
        shm = SharedMemory(name=shm_name, 
                            create=True, 
                            size=np.prod(param_shape)*float_size)
        # print("Shared Memory: ", shm_name, shm.size)
        # Now we need to create a numpy array in the shared memory
    shm_array = np.ndarray(param_shape, dtype=np.float64, buffer=shm.buf)
    shm_array[:] = params
    return shm, param_shape

def generate_large_circuits(large_block_circs: dict[str, list[Circuit]],
                            partitioned_circ: Circuit,
                            ens_size: int) -> list[Circuit]:
    all_circs = []
    num_digits = len(str(partitioned_circ.num_operations))
    for i in range(ens_size):
        circ = partitioned_circ.copy()
        for block_ind, (cycle, op) in enumerate(circ.operations_with_cycles()):
            pt = CircuitPoint(cycle, op.location[0])
            block_name = str(block_ind).zfill(num_digits)
            assert isinstance(op.gate, CircuitGate)
            assert isinstance(op.gate._circuit, Circuit)
            if block_name in large_block_circs:
                new_block_circ = large_block_circs[block_name][i]
                assert isinstance(new_block_circ, Circuit)
                if (op.gate._circuit.num_qudits != new_block_circ.num_qudits):
                    print("Mismatch in qudits: ", block_name, 
                            op.gate._circuit.num_qudits, 
                            new_block_circ.num_qudits)
                assert op.gate._circuit.num_qudits == new_block_circ.num_qudits
                circ.replace_with_circuit(pt, new_block_circ, 
                                          as_circuit_gate=True)
        all_circs.append(circ)
    return all_circs
# Circ 

if __name__ == '__main__':
    np.set_printoptions(precision=2, threshold=np.inf, linewidth=np.inf)
    max_tol = float(argv[1]) if len(argv) > 1 else 1.0
    use_noise = bool(int(argv[2])) if len(argv) > 2 else True

    # large_block_num = argv[2] if len(argv) > 2 else "00"
    large_block_nums = get_block_names(circ_name, extra="_tket")

    print("Large Block Names: ", large_block_nums)
    

    # Create shared memories
    all_shms: dict[str, dict[str, SharedMemory]] = {}
    all_param_shapes: dict[str, dict[str, tuple[int]]] = {}
    all_block_probs: dict[str, dict[str, np.ndarray]] = {}
    all_partitioned_circs: dict[str, Circuit] = {}
    initial_circs: dict[str, Circuit]= {}
    compiler = Compiler(num_workers=1)
    for large_block_num in large_block_nums:
        initial_circ_file = load_block(circ_name, large_block_num, extra="_tket")
        initial_circ: Circuit = Circuit.from_file(initial_circ_file)
        print("Original CX Count: ", initial_circ.count(CNOTGate()))
        initial_circs[large_block_num] = initial_circ

        print("Running LGT for block: ", large_block_num, max_tol)
        small_block_nums = get_sub_block_nums(large_block_num)
        print("Block Names: ", small_block_nums)
        sub_block_circs, small_partitioned_circ = partition_circs(compiler, 
                                                    large_block_num, 
                                                    small_block_nums)
        all_partitioned_circs[large_block_num] = small_partitioned_circ
        # block_targets = {}

        # Get sub-block data
        counts = []
        shms: dict[str, SharedMemory] = {}
        param_shapes = {}
        for small_block_num in small_block_nums:
            use, count = get_sub_block_count(large_block_num, 
                                            small_block_num, max_tol)
            orig_count = sub_block_circs[small_block_num].count(CNOTGate())
            if count >= orig_count or (not use):
                # We need to use the original count
                continue
            else:
                shm, param_shape = create_shared_memory(large_block_num,
                                                            small_block_num, 
                                                            max_tol)
                shms[small_block_num] = shm
                param_shapes[small_block_num] = param_shape

        if len(shms) == 0:
            print("Not using any ensembles for block: ", large_block_num)
            continue

        all_shms[large_block_num] = shms
        all_param_shapes[large_block_num] = param_shapes

        print("Created Shared Memories: ", shms.keys())


    compiler.close()
    ensemble_sizes = [1, 16, 64, 128, 256, 512, 1024]
    num_trials = 10
    for ens_size in ensemble_sizes:
        if use_noise:
            noise_text = "_noisy"
        else:
            noise_text = ""
        ens_data_file = f"ensemble_costs_{circ_name}_obs{noise_text}/{max_tol}_{ens_size}.pkl"
        if os.path.exists(ens_data_file):
            print("Already exists: ", ens_data_file)
            continue
        large_block_circs = {}
        for large_block_num in large_block_nums:
            block_circs = {}
            block_probs = {}
            initial_circ: Circuit = initial_circs[large_block_num]
            # initial_circ = load_circuit(circ_name, opt=False)
            small_block_nums = get_sub_block_nums(large_block_num)
            
            shms = all_shms[large_block_num]
            param_shapes = all_param_shapes[large_block_num]
            # Get all the circuits
            for small_block_num  in shms.keys():
                # print("Loading Circuits: ", small_block_num)
                circ_dir = f"{checkpoint_folder}_{large_block_num}_{max_tol}/block_{small_block_num}"
                full_path = f"{circ_dir}/ensemble_final.qasms"
                circs = load_ensemble(full_path)
                block_circs[small_block_num] = circs
                block_probs[small_block_num] = np.load(f"{circ_dir}/ensemble_final_probs.npy")
                num_unique_circs = param_shapes[small_block_num][0] * param_shapes[small_block_num][1]
                if len(block_probs[small_block_num]) != num_unique_circs:
                    print("Mismatch in number of circuits: ",
                            len(block_probs[small_block_num]), 
                            num_unique_circs)
                    # Default to uniform distribution
                    block_probs[small_block_num] = np.ones(num_unique_circs) / num_unique_circs
                small_partitioned_circ = all_partitioned_circs[large_block_num]

            ens = generate_small_block_circuits(block_circs, 
                                        small_block_nums,
                                        block_probs,
                                        param_shapes,
                                        shms=shms,
                                        pcirc=small_partitioned_circ,
                                        ens_size=ens_size * num_trials)
            
            qcircs = [get_qcirc(circ) for circ in ens]
            start = time.time()
            job_args = create_ham_args(qcircs, ham)
            transpile_time = time.time() - start

            num_shots = shots // ens_size

            start = time.time()
            results = [None] * len(job_args)
            # num_threads = 256 // (len(job_args))
            num_workers = min(len(job_args), 256)
            # Allocate number of cores per job arg
            num_threads = max(1, 256 // num_workers)
            print(f"Num Threads per Job: {num_threads} for {len(job_args)} jobs", flush=True)
            with ProcessPoolExecutor(max_workers=256) as executor:
                futures = {
                    executor.submit(run_single_nisq_circ, j, use_noise, 1024, num_threads): i
                    for i, j in enumerate(job_args)
                }

                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        results[idx] = future.result(timeout=600)
                    except Exception as e:
                        print(f"Task {idx} failed: {e}")
                        results[idx] = None
            # run_nisq_circs(job_args, max_parallel_threads=1)
            run_time = time.time() - start

            print(f"Num Circs: {len(qcircs)} Transpile Time: {transpile_time:.2f} s, Run Time: {run_time:.2f} s", flush=True)

    # Close all shared memories
    for large_block_num in all_shms.keys():
        shms = all_shms[large_block_num]
        for small_block_num in shms.keys():
            shm = shms[small_block_num]
            shm.close()
            shm.unlink()
        del all_shms[large_block_num]
