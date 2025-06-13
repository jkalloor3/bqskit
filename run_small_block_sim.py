from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import CNOTGate, CircuitGate, U3Gate, GlobalPhaseGate
from bqskit.qis import UnitaryMatrix
from sys import argv
import glob
import numpy as np
import os
import pickle
import cudaq

from pathlib import Path
from multiprocessing.shared_memory import SharedMemory

from bqskit.compiler import Compiler
from bqskit.passes import ScanPartitioner
from bqskit.ir.point import CircuitPoint

from bqskit.ir.lang.qasm2 import OPENQASM2Language

import csv

from common.io import (load_block, load_ensemble, get_block_names)

from sim_lib.sim_lib import (run_cudaq_nisq_circs)
from mpi4py import MPI

shots = 2 ** 15   
checkpoint_folder = "/pscratch/sd/j/jkalloor/bqskit/small_block_checkpoints_tket/{circ_name}"
checkpoint_folder_2 = "/pscratch/sd/j/jkalloor/bqskit/small_block_checkpoints_tket_2/{circ_name}"

lang = OPENQASM2Language()

base_bqskit_dir = "/pscratch/sd/j/jkalloor/bqskit"

def load_circuit(circ_name: str, timestep: int = 0, opt: bool = False) -> Circuit:
    if "JW" in circ_name:
        circ_name = f"JWCircs/{circ_name}.qasm"

    if opt:
        extra = "_tket"
    else:
        extra = ""
    
    file_name = f"{base_bqskit_dir}/ensemble_benchmarks{extra}/{circ_name}.qasm"
    if not os.path.exists(file_name):
        file_name = f"{base_bqskit_dir}/qce23_qfactor_benchmarks{extra}/{circ_name}.qasm"

    if not os.path.exists(file_name):
        file_name = f"{base_bqskit_dir}/ensemble_benchmarks_new{extra}/{circ_name}.qasm"

    return Circuit.from_file(filename=file_name)

def generate_full_circuits(block_circs: dict[str, list[Circuit]], 
                           block_names: list,
                           ens_size: int,
                           partitioned_circ: Circuit) -> list[Circuit]:
    all_circs = []
    for i in range(ens_size):
        circ = partitioned_circ.copy()
        ind = 0
        for cycle, op in circ.operations_with_cycles():
            pt = CircuitPoint(cycle, op.location[0])
            block_name = block_names[ind]
            ind += 1
            assert isinstance(op.gate, CircuitGate)
            assert isinstance(op.gate._circuit, Circuit)
            if block_name in block_circs:
                new_block_circ = block_circs[block_name][i]
                assert isinstance(new_block_circ, Circuit)
                assert op.gate._circuit.num_qudits == new_block_circ.num_qudits
                circ.replace_with_circuit(pt, new_block_circ, 
                                          as_circuit_gate=True)
        circ.unfold_all()
        all_circs.append(circ)
    return all_circs
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
        
def get_sub_block_nums(circ_name: str, block_num: str) -> list[str]:
    sub_block_path = f"{checkpoint_folder.format(circ_name=circ_name)}_{block_num}_*/block_*.data"
    sub_block_files = glob.glob(sub_block_path)
    sub_block_nums = set()
    for sub_block_file in sub_block_files:
        sub_block_num = Path(sub_block_file).name.split("_")[-1].split(".")[0]
        sub_block_nums.add(sub_block_num)
    sub_block_nums = sorted(list(sub_block_nums))
    return sub_block_nums

def get_sub_block_count(circ_name: str,
                        large_block_num: str, 
                        small_block_num: str, 
                        max_tol: float) -> tuple[bool, int]:
    qasm_file = f"{checkpoint_folder.format(circ_name=circ_name)}_{large_block_num}_{max_tol}/block_{small_block_num}/ensemble_final.qasms"
    csv_file = f"{checkpoint_folder.format(circ_name=circ_name)}_{large_block_num}_{max_tol}/block_{small_block_num}.csv"

    qasm_file_2 = f"{checkpoint_folder_2.format(circ_name=circ_name)}_{large_block_num}_{max_tol}/block_{small_block_num}/ensemble_final.qasms"
    csv_file_2 = f"{checkpoint_folder_2.format(circ_name=circ_name)}_{large_block_num}_{max_tol}/block_{small_block_num}.csv"
    # Read CSV file, if the ratio is < 20 then we can read counts
    if not os.path.exists(csv_file) and not os.path.exists(csv_file_2):
        print(f"CSV file does not exist: {csv_file}", flush=True)
        return False, 0
    
    def get_min_ratio(csv_file: str) -> float:
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            min_ratio = float("inf")
            final_frob_cost = float("inf")
            for row in reader:
                if "Ratio" in row:  # Check if the column value is not empty
                    min_ratio = min(min_ratio, float(row["Ratio"]))
                    final_frob_cost = min(final_frob_cost, float(row["Norm. Bias"]))
    
        return min_ratio
    
    min_ratio = get_min_ratio(csv_file)
        
    # Now we need to check if the ratio is less than 20
    if min_ratio > 10:
        min_ratio_2 = get_min_ratio(csv_file_2)
        if min_ratio_2 > 10:
            return False, 0
        else:
            qasm_file = qasm_file_2
            csv_file = csv_file_2
            min_ratio = min_ratio_2
    # Now we need to get the avg. number of CNOTs
    qasm_str = open(qasm_file, 'r').read()
    count = qasm_str.count("cx ")
    num_circs = qasm_str.count("BREAK") + 1
    count = count / num_circs
    return True, count

def get_random_circs(num_circs: int,
                    block_circs: list[Circuit],
                    probs: np.ndarray, 
                    shm: SharedMemory,
                    param_shape: tuple[int],
                    use_qp: bool = False) -> list[Circuit]:
    '''
    Generate a random circuit from the block_circs.
    '''
    assert len(block_circs) > 1
    # Now pick a random param from the shared memory
    shm_array = np.ndarray(param_shape, dtype=np.float64, buffer=shm.buf)

    # Else, we are using an ensemble
    if use_qp:
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
        circ_strs.append(rand_circ)

    return circ_strs

def aggregate_counts(counts: list[dict[str, int]]) -> dict[str, int]:
    '''
    Aggregate the counts from multiple circuits.
    '''
    aggregated_counts = {}
    for count in counts:
        for key, value in count.items():
            if key in aggregated_counts:
                aggregated_counts[key] += value
            else:
                aggregated_counts[key] = value
    return aggregated_counts


def generate_small_block_circuits(block_circs: dict[str, list[Circuit]], 
                           block_names: list,
                           block_probs: dict[str, np.ndarray],
                           param_shapes: dict[str, tuple[int]],
                           shms: dict[str, SharedMemory],
                           pcirc: Circuit,
                           ens_size: int,
                           use_qp: bool = False) -> list[Circuit]:
    all_circs = []

    circ_blocks = {}

    # target = pcirc.get_unitary()

    for block_name in shms.keys():
        shm = shms.get(block_name, None)
        param_shape = param_shapes.get(block_name, None)
        probs = block_probs.get(block_name, None)
        circ_blocks[block_name] = get_random_circs(ens_size,
                                                block_circs[block_name],
                                                probs=probs, 
                                                shm=shm,
                                                param_shape=param_shape,
                                                use_qp=use_qp)

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
        circ.unfold_all()
        # fix_phase(circ, target)
        # print("Dist: ", target.get_distance_from(circ.get_unitary()), 
        #       flush=True)
        all_circs.append(circ)
    # print("Finish Generating Circs: ", time.time() - start, flush=True)
    return all_circs

def create_shared_memory(circ_name: str,
                         large_block_num: str, 
                         small_block_num: str, 
                         max_tol: float,
                         mpi_rank: int = 0) -> tuple[SharedMemory, tuple[int]]:
    # Get params_file
    params_file = f"{checkpoint_folder.format(circ_name=circ_name)}_{large_block_num}_{max_tol}/block_{small_block_num}/ensemble_final_jiggle.npy"
    params: np.ndarray = np.load(params_file)
    param_shape = params.shape
    shm_name = f"{circ_name}_{large_block_num}_{small_block_num}_{max_tol}_{mpi_rank}"
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


if __name__ == '__main__':
    cudaq.set_target('nvidia')
    np.set_printoptions(precision=2, threshold=np.inf, linewidth=np.inf)
    circ_name = argv[1]
    max_tol = float(argv[2]) if len(argv) > 1 else 1.0
    use_noise = bool(int(argv[3])) if len(argv) > 2 else True
    use_qp = bool(int(argv[4])) if len(argv) > 3 else False

    partitioned_circ = pickle.load(open(f"/pscratch/sd/j/jkalloor/bqskit/partitioned_circs/{circ_name}.pickle", "rb"))

    if use_noise:
        COHERENT_ERROR = 1e-4  # Coherent error to add to the circuits
    else:
        COHERENT_ERROR = 0.0

    # large_block_num = argv[2] if len(argv) > 2 else "00"
    large_block_nums = get_block_names(circ_name, extra="_tket")

    # full_circ = Circuit.from_file(f"ensemble_benchmarks/{circ_name}.qasm")
    full_circ = load_circuit(circ_name, opt=False)

    num_processes = MPI.COMM_WORLD.Get_size()
    mpi_rank = MPI.COMM_WORLD.Get_rank()

    full_circ.unfold_all()
    target_obs_mag = run_cudaq_nisq_circs([full_circ],
                                          ham=None,
                                          add_coherent_error=COHERENT_ERROR,
                                           use_noise=use_noise, 
                                           num_shots=shots,
                                           average=False)
    print("Target Results: ", target_obs_mag)
    print("Large Block Names: ", large_block_nums)

    # Create shared memories
    all_shms: dict[str, dict[str, SharedMemory]] = {}
    all_param_shapes: dict[str, dict[str, tuple[int]]] = {}
    all_block_circs: dict[str, dict[str, list[Circuit]]] = {}
    all_block_probs: dict[str, dict[str, np.ndarray]] = {}
    all_partitioned_circs: dict[str, Circuit] = {}
    initial_circs: dict[str, Circuit]= {}
    if mpi_rank == 0:
        compiler = Compiler(num_workers=1)
    for large_block_num in large_block_nums:
        block_circs = {}
        block_probs = {}
        block_inds = {}
        initial_circ_file = load_block(circ_name, large_block_num, extra="_tket")
        initial_circ: Circuit = Circuit.from_file(initial_circ_file)
        print("Original CX Count: ", initial_circ.count(CNOTGate()))
        initial_circs[large_block_num] = initial_circ
        small_block_nums = get_sub_block_nums(circ_name, large_block_num)
        if mpi_rank == 0:
            sub_block_circs, small_partitioned_circ = partition_circs(compiler, 
                                                        large_block_num, 
                                                        small_block_nums)
        else:
            sub_block_circs = None
            small_partitioned_circ = None
        small_partitioned_circ: Circuit = MPI.COMM_WORLD.bcast(small_partitioned_circ, root=0)
        sub_block_circs: list[Circuit] = MPI.COMM_WORLD.bcast(sub_block_circs, root=0)
        all_partitioned_circs[large_block_num] = small_partitioned_circ
        # block_targets = {}

        # Get sub-block data
        counts = []
        shms: dict[str, SharedMemory] = {}
        param_shapes = {}
        for small_block_num in small_block_nums:
            use, count = get_sub_block_count(circ_name, large_block_num, 
                                            small_block_num, max_tol)
            orig_count = sub_block_circs[small_block_num].count(CNOTGate())
            if (count >= orig_count and use_noise) or (not use):
                # We need to use the original count
                continue
            else:
                shm, param_shape = create_shared_memory(circ_name, large_block_num,
                                                            small_block_num, 
                                                            max_tol,
                                                            mpi_rank=mpi_rank)
                shms[small_block_num] = shm
                param_shapes[small_block_num] = param_shape

        if len(shms) == 0:
            print("Not using any ensembles for block: ", large_block_num)
            continue

        all_shms[large_block_num] = shms
        all_param_shapes[large_block_num] = param_shapes

        print("Created Shared Memories: ", shms.keys())

        # Get all the circuits
        for small_block_num  in shms.keys():
            print("Loading Circuits: ", small_block_num)
            circ_dir = f"{checkpoint_folder.format(circ_name=circ_name)}_{large_block_num}_{max_tol}/block_{small_block_num}"
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

        all_block_circs[large_block_num] = block_circs
        all_block_probs[large_block_num] = block_probs
        print("Waiting at Barrier, ", mpi_rank, flush=True)
        MPI.COMM_WORLD.barrier()

    if mpi_rank == 0:
        compiler.close()
        print("Closed Compiler", flush=True)
    ensemble_sizes = [1, 4, 8, 64, 128, 256, 512, 1024]
    num_trials = 8
    for ens_size in ensemble_sizes:
        if use_noise:
            noise_text = "_noisy"
        else:
            noise_text = ""

        if use_qp:
            noise_text += "_qp"

        ens_data_file = f"ensemble_costs_{circ_name}_obs{noise_text}_mpi/{max_tol}_{ens_size}_{mpi_rank}.pkl"
        if os.path.exists(ens_data_file):
            print("Already exists: ", ens_data_file)
            continue
        large_block_circs = {}
        for large_block_num in large_block_nums:
            if large_block_num not in shms:
                print("Skipping large block: ", large_block_num, 
                      " as no small blocks are available")
                continue
            # print("Running LGT for block: ", large_block_num, max_tol)
            initial_circ: Circuit = initial_circs[large_block_num]
            # initial_circ = load_circuit(circ_name, opt=False)
            # print("Original CX Count: ", initial_circ.count(CNOTGate()))
            small_block_nums = get_sub_block_nums(circ_name, large_block_num)
            # print("Block Names: ", small_block_nums)
            
            shms = all_shms[large_block_num]
            param_shapes = all_param_shapes[large_block_num]
            block_circs = all_block_circs[large_block_num]
            block_probs = all_block_probs[large_block_num]
            small_partitioned_circ = all_partitioned_circs[large_block_num]

            print("LOADED CIRCUITS", flush=True)

            ens = generate_small_block_circuits(block_circs, 
                                        small_block_nums,
                                        block_probs,
                                        param_shapes,
                                        shms=shms,
                                        pcirc=small_partitioned_circ,
                                        ens_size=(ens_size * num_trials // num_processes),
                                        use_qp=use_qp)
            
            large_block_circs[large_block_num] = ens
            print(f"Generated {len(ens)} circuits for large block {large_block_num}", flush=True)

        num_shots = shots // ens_size
        full_ens = generate_full_circuits(large_block_circs,
                                          large_block_nums,
                                          ens_size=(ens_size * num_trials // num_processes),
                                          partitioned_circ=partitioned_circ)
        

        print(f"Generated {len(full_ens)} full circuits for ensemble size {ens_size}", flush=True)

        all_mags = run_cudaq_nisq_circs(circs=full_ens, 
                                        ham=None, 
                                        add_coherent_error=COHERENT_ERROR, 
                                        use_noise=use_noise, 
                                        num_shots=num_shots, 
                                        average=False)
        sub_counts = [all_mags[i * ens_size:(i + 1) * ens_size] for i in range(num_trials // num_processes)]
        ensemble_mags = [aggregate_counts(counts) for counts in sub_counts]
        Path(ens_data_file).parent.mkdir(parents=True, exist_ok=True)
        pickle.dump((ensemble_mags), open(ens_data_file, "wb"))

    # Close all shared memories
    for large_block_num in all_shms.keys():
        shms = all_shms[large_block_num]
        for small_block_num in shms.keys():
            shm = shms[small_block_num]
            shm.close()
            shm.unlink()
        del all_shms[large_block_num]
