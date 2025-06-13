from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import CNOTGate, CircuitGate
from sys import argv
import glob
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt

from pathlib import Path
from multiprocessing.shared_memory import SharedMemory

from bqskit.compiler import Compiler
from bqskit.passes import ScanPartitioner
from bqskit.ir.gates import GlobalPhaseGate
from bqskit.ir.point import CircuitPoint

from bqskit.ir.lang.qasm2 import OPENQASM2Language

import csv
from util import (load_block, load_ensemble, load_avg_ensemble_counts_full,
                  get_block_names)
from util.distance import (normalized_gp_frob_cost, trace_distance, tvd, 
                           get_density_matrix, get_average_density_matrix)
from util.fix_global_phase import fix_phase

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from bqskit.qis import UnitaryMatrix
from bqskit.ext import bqskit_to_qiskit

import time

shots = 100
circ_name = "shor_12_w_qft"
checkpoint_folder = f"small_block_checkpoints_final_paper_4_more_cx_tket/{circ_name}"

USE_QP = True

lang = OPENQASM2Language()

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
    print("Final Frobenius Cost: ", final_frob_cost)
    print("Min Ratio: ", min_ratio)
    # Now we need to get the avg. number of CNOTs
    count = load_avg_ensemble_counts_full(qasm_file,
                                          None,
                                          None,
                                          target_error=max_tol,
                                          count_t=False,
                                          count_rz=False)
    return True, count

def get_qcirc(circ: Circuit) -> QuantumCircuit:
    circ.unfold_all()
    # Remove all GlobalPhaseGates
    circ.remove_all(GlobalPhaseGate())
    return bqskit_to_qiskit(circ)

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

def generate_small_block_circuits(block_circs: dict[str, list[Circuit]], 
                           block_names: list,
                           block_probs: dict[str, np.ndarray],
                           param_shapes: dict[str, tuple[int]],
                           shms: dict[str, SharedMemory],
                           pcirc: Circuit,
                           ens_size: int) -> list[Circuit]:
    all_circs = []

    circ_blocks = {}

    start = time.time()
    for block_name in shms.keys():
        shm = shms.get(block_name, None)
        param_shape = param_shapes.get(block_name, None)
        probs = block_probs.get(block_name, None)
        circ_blocks[block_name] = get_random_circs(ens_size,
                                                block_circs[block_name],
                                                probs=probs, 
                                                shm=shm,
                                                param_shape=param_shape)
        
    # print("Finish Getting Circs: ", time.time() - start, flush=True)
    start = time.time()
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
    max_tol = float(argv[1]) if len(argv) > 1 else 3.0
    # large_block_num = argv[2] if len(argv) > 2 else "00"
    large_block_nums = get_block_names(circ_name, extra="_tket")
    partitioned_circ_file = f"partitioned_circs/{circ_name}.pickle"
    partitioned_circ = pickle.load(open(partitioned_circ_file, "rb"))
    # large_block_nums = ["11"]

    full_circ = partitioned_circ.copy()
    full_circ.unfold_all()
    target = full_circ.get_unitary()
    target_sv = Statevector.from_instruction(get_qcirc(full_circ)).data
    target_sv_prob = np.abs(target_sv)**2
    target_dm = get_density_matrix(target_sv)

    print("Calculated Target data")

    print("Large Block Names: ", large_block_nums)
    print("Large Block Qudit Counts: ",[b.gate._circuit.num_qudits for b in partitioned_circ.operations()])

    ensemble_sizes = [1, 2, 4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    num_trials = 10
    for ens_size in ensemble_sizes:
        ens_data_file = f"ensemble_costs_{circ_name}/{max_tol}_{ens_size}.pkl"
        if os.path.exists(ens_data_file):
            print("Already exists: ", ens_data_file)
            continue
        large_block_circs = {}
        for large_block_num in large_block_nums:
            block_circs = {}
            small_block_nums = []
            block_probs = {}
            block_inds = {}
            block_targets = {}
            img_file = f"ensemble_costs_shor/{circ_name}_{large_block_num}_{max_tol}_qp_log.png"
            if os.path.exists(img_file):
                print("Already exists: ", large_block_num, max_tol)
                continue
            print("Running Shor for block: ", large_block_num, max_tol)
            initial_circ_file = load_block(circ_name, large_block_num, extra="_tket")
            initial_circ: Circuit = Circuit.from_file(initial_circ_file)
            # initial_circ = load_circuit(circ_name, opt=False)
            print("Original CX Count: ", initial_circ.count(CNOTGate()))
            small_block_nums = get_sub_block_nums(large_block_num)
            print("Block Names: ", small_block_nums)
            compiler = Compiler(num_workers=1)
            sub_block_circs, small_partitioned_circ = partition_circs(compiler, 
                                                        large_block_num, 
                                                        small_block_nums)
            block_targets = {}

            # Get sub-block data
            counts = []
            shms: dict[str, SharedMemory] = {}
            param_shapes = {}
            for small_block_num in small_block_nums:
                use, count = get_sub_block_count(large_block_num, 
                                                small_block_num, max_tol)
                orig_count = sub_block_circs[small_block_num].count(CNOTGate())
                if count >= orig_count:
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

            print("Created Shared Memories: ", shms.keys())

            # return

            # Get all the circuits
            for small_block_num  in shms.keys():
                print("Loading Circuits: ", small_block_num)
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

            # print("Num Circuits: ", [len(ens) for ens in block_ensembles.values()], flush=True)

            print("LOADED CIRCUITS", flush=True)

            ens = generate_small_block_circuits(block_circs, 
                                        small_block_nums,
                                        block_probs,
                                        param_shapes,
                                        shms=shms,
                                        # block_targets=block_targets,
                                        pcirc=small_partitioned_circ,
                                        ens_size=ens_size * num_trials)
            
            print(f"Generated {len(ens)} circuits for large block {large_block_num}", flush=True)

            for small_block_num, shm in shms.items():
                shm.close()
                shm.unlink()

            large_block_circs[large_block_num] = ens

        # Now, we need to put together the circuits

        ens = generate_large_circuits(large_block_circs,
                                partitioned_circ,
                                ens_size=ens_size * num_trials)
        
        # Calculate the costs
        ensemble_frob_costs = []
        ensemble_tvd_costs = []
        ensemble_trace_dist_costs = []
        for i in range(num_trials):
            sub_ens = ens[i * ens_size:(i + 1) * ens_size]
            qcircs = [get_qcirc(circ) for circ in sub_ens]
            ens_svs = np.array([Statevector.from_instruction(circ).data for circ in qcircs])
            # mean_sv = np.mean(ens_svs, axis=0)
            # Get the average density matrix
            avg_dm = get_average_density_matrix(ens_svs)
            ens_sv_probs = [np.abs(sv)**2 for sv in ens_svs]
            mean_sv_prob = np.mean(ens_sv_probs, axis=0)
            # Get the costs
            uns = [c.get_unitary() for c in sub_ens]
            mean_un = np.mean(uns, axis=0)
            frob_cost = normalized_gp_frob_cost(mean_un, target)
            trace_dist_cost = trace_distance(avg_dm, target)
            tvd_cost = tvd(mean_sv_prob, target_sv_prob)
            ensemble_frob_costs.append(frob_cost)
            ensemble_trace_dist_costs.append(trace_dist_cost)
            ensemble_tvd_costs.append(tvd_cost)
            
        # Save data
        Path(ens_data_file).parent.mkdir(parents=True, exist_ok=True)
        pickle.dump((ensemble_frob_costs,
                     ensemble_trace_dist_costs,
                     ensemble_tvd_costs), open(ens_data_file, "wb"))
