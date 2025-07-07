from bqskit.ir.circuit import Circuit, CircuitGate, CircuitPoint
from bqskit.ir.gates import U3Gate, CNOTGate
import time
import numpy as np
from pathlib import Path
import pickle
from sys import argv
import os
from typing import Generator
from mpi4py import MPI
import cudaq
from itertools import chain

from bqskit.compiler import Compiler
from bqskit.passes import ScanPartitioner


from common.io import (load_block, get_block_names)
from util import  get_sub_block_nums, get_file_names, load_jiggled_ensemble, EnsembleSampler, get_sub_block_count, GateCounter, load_circuit, tvd_dict
from sim_lib import run_cudaq_nisq_circs

cliff_t = False
if cliff_t:
    # print("Using cliff-t circuits", flush=True)
    base_checkpoint_dir = "small_block_checkpoints_final_paper_4_clifft_tket"
    partitioned_data_file = "partitioned_data_all_clifft_circs.pickle"
else:
    # print("Using non-cliff-t circuits", flush=True)
    base_checkpoint_dir = "small_block_checkpoints_final_paper_4_more_cx_tket"
    partitioned_data_file = "partitioned_data_all_circs.pickle"

tol = 3.0
base_dir_form = os.path.join(base_checkpoint_dir, "{circ_name}_{large_block_num}_" + f"{tol}/")
# Get partitioned circuits for all 8-qubit blocks


def random_init(circs: list[Circuit]) -> list[list[Circuit]]:
    '''
    # Take a list of circuits of size n and returns a list of circuits 
    # of size 3 x n, where each circuit is initialized with a random state.
    
    circs: List of circuits to initialize.
    '''
    np.random.seed(42)  # For reproducibility
    num_qudits = circs[0].num_qudits

    # Generate random initial states for each circuit
    init_circs = []
    for _ in range(3):
        init_circ = Circuit(num_qudits=num_qudits)
        # Initialize with random U3 gates on every qubit
        for q in range(num_qudits):
            init_circ.append_gate(U3Gate(), (q,), params=(np.random.rand(3) * 2 * np.pi))

        # Add n / 2 CNOT gates between random pairs of qubits
        for _ in range(num_qudits // 2):
            q1, q2 = np.random.choice(num_qudits, size=2, replace=False)
            init_circ.append_gate(CNOTGate(), (q1, q2))
            # U3s to these qubits as well
            init_circ.append_gate(U3Gate(), (q1,), params=(np.random.rand(3) * 2 * np.pi))
            init_circ.append_gate(U3Gate(), (q2,), params=(np.random.rand(3) * 2 * np.pi))

        
        init_circs.append(init_circ)

    # Now replace every circuit with the init circuits + circuit
    random_circs = [[init_circ + circ for circ in circs] for init_circ in init_circs]
    return random_circs


def partition_circs(compiler: Compiler,
                    circ_name, 
                    block_num: str) -> tuple[dict[str, Circuit], Circuit]:
    workflow = [
        ScanPartitioner(4)
    ]
    circ_file = load_block(circ_name, block_num, extra="_tket")
    circ = Circuit.from_file(circ_file)
    return compiler.submit(circ, workflow=workflow)

def get_good_blocks(circ_name, tol, cliff_t,
                    checkpoint_folder_form: str,
                    partitioned_data: dict) -> dict[str, set[str]]:
    '''
    Get the good blocks for each circuit from the checkpoint folder.
    
    checkpoint_folder_form: The form of the checkpoint folder path.
    '''
    good_blocks = {}
    counter = GateCounter(est=False, cache_file=None)
    max_ratio = min(20, (10 ** (tol) / 4))
    num_good_blocks = 0
    for large_block_num in get_block_names(circ_name, extra="_tket"):
        good_blocks[large_block_num] = set()
        small_block_circs, _ = partitioned_data[large_block_num]
        large_block_dir = checkpoint_folder_form.format(large_block_num=large_block_num)
        # print(f"Large Block Dir: {large_block_dir}", flush=True)
        for small_block_num, small_circ in small_block_circs.items():
            # print(f"Small Block: {small_block_num} in {large_block_num}", flush=True)
            good, count = get_sub_block_count(large_block_dir, 
                                                small_block_num, tol, cliff_t)
            if cliff_t:
                original_count = counter.count_t(small_circ, target_error=(10 ** (- 2 * tol) * max_ratio))
            else:
                # Count CNOTs in the circuit
                original_count = counter.count_cx(small_circ)
            if not good:
                continue
            elif count > original_count:
                continue
            else:
                # Use block
                num_good_blocks += 1
                good_blocks[large_block_num].add(small_block_num)
        if len(good_blocks[large_block_num]) == 0:
            # print(f"No good blocks found for circuit {circ_name} in large block {large_block_num}", flush=True)
            good_blocks.pop(large_block_num)
    # print(f"Found {num_good_blocks} good blocks for circuit {circ_name}", flush=True)
    return good_blocks

def generate_large_block_circ(circ_name: str,
                              large_block_num: str,
                              p_circ: Circuit,
                              good_blocks: set[str]) -> Generator[Circuit, None, None]:
    circ_samplers = {}
    for good_block in good_blocks:
        files = get_file_names(
            base_dir_form.format(circ_name=circ_name, large_block_num=large_block_num),
            good_block
        )
        all_circ_params = load_jiggled_ensemble(*files[:-1])
        circ_samplers[good_block] = EnsembleSampler(all_circ_params, 
                                                    cliff_t=cliff_t)
        

    while True:
        block_circ = p_circ.copy()
        # print(good_blocks, flush=True)
        small_num_digits = len(str(block_circ.num_operations))
        i = 0
        for cycle, op in block_circ.operations_with_cycles():
            assert isinstance(op.gate, CircuitGate), "Operation is not a CircuitGate"
            small_block_num = str(i).zfill(small_num_digits)
            i += 1
            if small_block_num not in good_blocks:
                continue
            # Sample a small block circuit
            # print("Sampling small block:", small_block_num, flush=True)
            small_circ = next(circ_samplers[small_block_num])
            pt = CircuitPoint(cycle, op.location[0])
            block_circ.replace_with_circuit(pt, small_circ, as_circuit_gate=True)
        block_circ.unfold_all()
        yield block_circ


def generate_full_circ(circ_name: str, 
                       big_partitioned_circ: Circuit,
                       good_blocks: dict[str, set[str]],
                       partitioned_data: dict[str, Circuit]) -> Generator[Circuit, 
                                                                          None, 
                                                                          None]:
    '''
    Generate the full circuit for a given circuit name and block number.

    circ_name: The name of the circuit.
    cliff_t: Whether to use cliff-t circuits or not.
    big_partitioned_circ: The circuit that has been partitioned into blocks.
    good_blocks: A dictionary containing the good blocks for each circuit. It is 
    a mapping from large block number to a set of small block numbers
    partitioned_data: A dictionary containing the partitioned circuit for each large
    block.



    '''
    # Large Block Circ generators
    large_circ_generators = {}
    i = 0
    large_block_num_digits = len(str(big_partitioned_circ.num_operations))
    for cycle, op in big_partitioned_circ.operations_with_cycles():
        assert isinstance(op.gate, CircuitGate), "Operation is not a CircuitGate"
        large_block_num = str(i).zfill(large_block_num_digits)
        i += 1
        if large_block_num not in good_blocks:
            continue
        large_circ_generators[large_block_num] = generate_large_block_circ(
            circ_name,
            large_block_num,
            partitioned_data[large_block_num][1],
            good_blocks[large_block_num]
        )

    while True:
        out_circ = big_partitioned_circ.copy()
        i = 0
        for cycle, op in out_circ.operations_with_cycles():
            assert isinstance(op.gate, CircuitGate), "Operation is not a CircuitGate"
            large_block_num = str(i).zfill(large_block_num_digits)
            i += 1
            if large_block_num not in good_blocks:
                continue
            sample_block_circ = next(large_circ_generators[large_block_num])
            pt = CircuitPoint(cycle, op.location[0])
            out_circ.replace_with_circuit(pt, sample_block_circ, as_circuit_gate=True)
        
        out_circ.unfold_all()
        yield out_circ


if __name__ == "__main__":
    # circ_names = ["qae11"]
    circ_name = argv[1]
    tol = float(argv[2])
    use_noise = bool(int(argv[3])) if len(argv) > 3 else False
    # compiler = Compiler(num_workers=len(circ_names))
    cliff_t = False

    # missing_circ_names = circ_names.copy()
    if os.path.exists(partitioned_data_file):
        with open(partitioned_data_file, "rb") as f:
            all_partitioned_data = pickle.load(f)
        # print("Loaded partitioned data from file.", list(all_partitioned_data.keys()))
        # Only run on circs that are not in data
        missing_circ_names = [circ_name] if circ_name not in all_partitioned_data else []
        # print(f"Missing circ names: {missing_circ_names}")

    total_circs_queried = 0

    start = time.time()

    mpi_rank = MPI.COMM_WORLD.Get_rank()
    mpi_size = MPI.COMM_WORLD.Get_size()

    print(f"MPI Rank: {mpi_rank}, MPI Size: {mpi_size}", flush=True)

    # print(f"MPI Rank: {mpi_rank}, MPI Size: {mpi_size}", flush=True)
    cudaq.set_target('nvidia')
    
    ens_sizes = [1, 10, 100, 1000, 10000]
    total_shots = 1024 * max(ens_sizes)

    num_trials = 3

    # for circ_name in circ_names:
    i = 0
    full_circ = load_circuit(circ_name)
    full_circ.remove_all_measurements()
    # print(f"Running full circuit {circ_name} with {full_circ.num_qudits} qudits and {full_circ.num_operations} operations", flush=True)
    random_full = random_init([full_circ])
    random_full = list(chain.from_iterable(random_full))
    true_dist: list[dict] = run_cudaq_nisq_circs(circs=random_full,
                                        ham=None,
                                        use_noise=False,
                                        num_shots=total_shots,
                                        average=False)
    

    if use_noise:
        noisy_tvd_file = f"ensemble_tvds/{circ_name}_base_noisy.pkl"
        if not os.path.exists(noisy_tvd_file) and mpi_rank == 0:
            noisy_dist: list[dict] = run_cudaq_nisq_circs(circs=random_full,
                                            ham=None,
                                            use_noise=True,
                                            num_shots=total_shots,
                                            average=False)
            noisy_tvds = [tvd_dict(true_dist[i], noisy_dist[i]) for i in range(len(true_dist))]
            print(f"Noisy TVDs for {circ_name}: {noisy_tvds}", flush=True)
            Path("ensemble_tvds").mkdir(parents=True, exist_ok=True)
            with open(noisy_tvd_file, "wb") as f:
                pickle.dump(noisy_tvds, f)

    assert sum(true_dist[0].values()) == total_shots, "Total shots in true distribution does not match total shots requested"
    
    checkpoint_folder_form = f"{base_checkpoint_dir}/{circ_name}" + "_{large_block_num}_" + f"{tol}/"
    partitioned_circ_file=f"partitioned_circs/{circ_name}.pickle"
    partitioned_circ = pickle.load(open(partitioned_circ_file, "rb"))
    full_circ_generator = generate_full_circ(
        circ_name=circ_name,
        big_partitioned_circ=partitioned_circ,
        good_blocks=get_good_blocks(circ_name=circ_name,
                                    tol=tol,
                                    cliff_t=cliff_t,
                                    checkpoint_folder_form=checkpoint_folder_form,
                                    partitioned_data=all_partitioned_data[circ_name]),
        partitioned_data=all_partitioned_data[circ_name]
    )

    for ens_size in ens_sizes:
        noise_text = "_noisy" if use_noise else ""
        output_file = f"ensemble_tvds/{circ_name}_{ens_size}_{tol}{noise_text}.pkl"
        if os.path.exists(output_file):
            continue
        ens = [next(full_circ_generator) for _ in range(ens_size * num_trials)]
        # Use 3 different initial states for each member of ensemble
        random_ens = random_init(ens)
        assert len(random_ens) == len(random_full), f"Random ensemble size does not match full ensemble size: {len(random_ens)} != {len(random_full)}"
        ens_shots = total_shots // ens_size
        # print(f"Running ensemble of {len(ens)} circuits with {ens_shots} shots each", flush=True)
        rand_tvds: list[list[float]] = []
        for rand_ind, rand_ens in enumerate(random_ens):
            assert len(rand_ens) == num_trials * ens_size
            results : list[dict] = run_cudaq_nisq_circs(circs=rand_ens,
                            ham=None,
                            use_noise=use_noise,
                            num_shots=ens_shots,
                            average=False)
            # Combine all results of type SampleResult
            # Get one total counts per num_trials
            tvds = []
            for trial in range(num_trials):
                total_counts = {}
                for res in results[trial * ens_size:(trial + 1) * ens_size]:
                    for key, value in res.items():
                        if key not in total_counts:
                            total_counts[key] = 0
                        total_counts[key] += value

                total_sum = sum(total_counts.values())
                assert total_sum == total_shots, "Total shots in ensemble does not match total shots requested"

                tvd = tvd_dict(true_dist[rand_ind], total_counts)
                tvds.append(tvd)
            rand_tvds.append(tvds)
        
        # MPI gather all the tvds
        all_tvds: list[list[list[float]]] = MPI.COMM_WORLD.gather(rand_tvds, root = 0)
        if mpi_rank == 0:
            # Flatten the list of TVDs
            joined_tvds = all_tvds[0]
            for mpi_tvds in all_tvds[1:]:
                for rand_ind, tvd in enumerate(mpi_tvds):
                    joined_tvds[rand_ind].extend(tvd)

            print(joined_tvds, flush=True)

        # Save all TVDs to a file
        if mpi_rank == 0:
            Path("ensemble_tvds").mkdir(parents=True, exist_ok=True)
            with open(output_file, "wb") as f:
                pickle.dump(joined_tvds, f)
            print(f"Saved TVDs for {circ_name} with ensemble size {ens_size} to {output_file}", flush=True)
        
    end = time.time()
    if mpi_rank == 0:
        print(f"Time taken: {end - start} seconds")

    