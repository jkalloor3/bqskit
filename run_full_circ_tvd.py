from bqskit.ir.circuit import Circuit, CircuitGate, CircuitPoint
from bqskit.qis import StateVector, UnitaryMatrix
import time
import numpy as np
from pathlib import Path
import pickle
from sys import argv
import os
from typing import Generator

from bqskit.compiler import Compiler

from util import  (get_file_names, load_jiggled_ensemble, EnsembleSampler, 
                   get_sub_block_count, GateCounter, load_circuit, trace_distance,
                   get_density_matrix, get_block_names, frobenius_cost, tvd_dict)

from bqskit.runtime import get_runtime


from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData

cliff_t = False

NUM_RANDOM_SEEDS = 10
NUM_TOTAL_SHOTS = 64000 * 10

if cliff_t:
    # print("Using cliff-t circuits", flush=True)
    base_checkpoint_dir = "small_block_checkpoints_final_paper_4_clifft_tket"
    partitioned_data_file = "partitioned_data_all_clifft_circs.pickle"
else:
    # print("Using non-cliff-t circuits", flush=True)
    base_checkpoint_dir = "small_block_checkpoints_final_paper_4_more_cx_tket"
    partitioned_data_file = "partitioned_data_all_circs.pickle"

base_dir_form = os.path.join(base_checkpoint_dir, "{circ_name}_{large_block_num}_" + "{tol}/")
# Get partitioned circuits for all 8-qubit blocks

def get_shots(probs: np.ndarray, total_shots: int) -> dict[str, int]:
    '''
    Get the shots for a given probability distribution.

    Each index corresponds to a bitstring in lexicographical order.
    '''
    states = [bin(i)[2:].zfill(int(np.log2(len(probs)))) for i in range(len(probs))]
    counts = np.random.multinomial(total_shots, probs)
    return {states[i]: counts[i] for i in range(len(states)) if counts[i] > 0}

def sum_counts(counts_list: list[dict[str, int]]) -> dict[str, int]:
    '''
    Sum the counts from a list of count dictionaries.
    '''
    total_counts = {}
    for counts in counts_list:
        for state, count in counts.items():
            if state in total_counts:
                total_counts[state] += count
            else:
                total_counts[state] = count
    return total_counts


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
    print(f"Found {num_good_blocks} good blocks for circuit {circ_name}", flush=True)
    return good_blocks

def generate_large_block_circ(circ_name: str,
                              large_block_num: str,
                              tol: float,
                              p_circ: Circuit,
                              good_blocks: set[str]) -> Generator[UnitaryMatrix, None, None]:
    circ_samplers = {}
    for good_block in good_blocks:
        files = get_file_names(
            base_dir_form.format(circ_name=circ_name, large_block_num=large_block_num, tol=tol),
            good_block
        )
        all_circ_params = load_jiggled_ensemble(*files[:-1])
        circ_samplers[good_block] = EnsembleSampler(all_circ_params, 
                                                    cliff_t=cliff_t)

    while True:
        block_circ = p_circ.copy()
        small_num_digits = len(str(p_circ.num_operations))
        i = 0
        for cycle, op in p_circ.operations_with_cycles():
            assert isinstance(op.gate, CircuitGate), "Operation is not a CircuitGate"
            small_block_num = str(i).zfill(small_num_digits)
            i += 1
            if small_block_num in good_blocks:
                # Sample a small block circuit
                pt = CircuitPoint(cycle, op.location[0])
                small_circ = next(circ_samplers[small_block_num])
                block_circ.replace_with_circuit(pt, 
                                                small_circ, 
                                                as_circuit_gate=True)
        yield block_circ

def generate_full_circ(circ_name: str, 
                       big_partitioned_circ: Circuit,
                       tol: float,
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
            tol,
            partitioned_data[large_block_num][1],
            good_blocks[large_block_num]
        )

    while True:
        out_circ = big_partitioned_circ.copy()
        i = 0
        for cycle, op in big_partitioned_circ.operations_with_cycles():
            assert isinstance(op.gate, CircuitGate), "Operation is not a CircuitGate"
            large_block_num = str(i).zfill(large_block_num_digits)
            i += 1
            if large_block_num in good_blocks:
                sample_circ = next(large_circ_generators[large_block_num])
                pt = CircuitPoint(cycle, op.location[0])
                out_circ.replace_with_circuit(pt, 
                                              sample_circ, 
                                              as_circuit_gate=True)

        yield out_circ

class FullCircTDPass(BasePass):
    '''
    A pass that runs the full circuit trace distance calculation.
    '''

    def __init__(self, 
                 circ_name: str, 
                 tol: float,
                 num_qudits: int,
                 ens_sizes: list[int],
                 partitioned_circ: Circuit,
                 checkpoint_folder_form: str,
                 all_partitioned_data: dict[str, Circuit],
                 cliff_t: bool = False) -> None:
        super().__init__()
        self.circ_name = circ_name
        self.tol = tol
        self.ens_sizes = ens_sizes
        self.num_qudits = num_qudits
        self.rand_svs = [StateVector.random(num_qudits) for _ in range(NUM_RANDOM_SEEDS)]
        self.partitioned_circ = partitioned_circ
        self.good_blocks = get_good_blocks(circ_name=circ_name,
                                           tol=tol,
                                           cliff_t=cliff_t,
                                           checkpoint_folder_form=checkpoint_folder_form,
                                           partitioned_data=all_partitioned_data[circ_name])
        self.partitioned_data = all_partitioned_data[circ_name]
        self.num_trials = 6


    async def get_trial_counts(self, 
                               ens_size: int, 
                               shots: int) -> list[dict[str, int]]:
        '''
        Get the average unitary for a given ensemble size.
        '''
        full_circ_generator = generate_full_circ(
            circ_name=self.circ_name,
            big_partitioned_circ=self.partitioned_circ,
            tol=self.tol,
            good_blocks=self.good_blocks,
            partitioned_data=self.partitioned_data
        )
        all_shot_data = [{} for _ in range(len(self.rand_svs))]
        for _ in range(ens_size):
            circ = next(full_circ_generator)
            # Calculate shots based on random SVs
            sv_outs = [circ.get_statevector(sv) for sv in self.rand_svs]
            probs = [sv.get_probs() for sv in sv_outs]
            # Get counts for each sv
            counts = [get_shots(p, shots) for p in probs]
            all_shot_data = [sum_counts([all_shot_data[i], 
                                         counts[i]]) for i in range(len(self.rand_svs))]
        return all_shot_data
    
    async def get_full_counts_outer(self, ens_size: int) -> dict[str, int]:
        '''
        Get the average unitary for a given ensemble size.
        '''

        # Split ens_size into chunks of 100
        CHUNK_SIZE = 100
        if ens_size > CHUNK_SIZE:
            split_ens = [CHUNK_SIZE] * (ens_size // CHUNK_SIZE)
        else:
            split_ens = [ens_size]

        shots_per_circ = NUM_TOTAL_SHOTS // ens_size

        print("Shots per circ: ", shots_per_circ, flush=True)

        # M x N array of dicts
        all_shot_data: list[list[dict[str, int]]] = await get_runtime().map(self.get_trial_counts, 
                                                                      split_ens, 
                                                                      shots=shots_per_circ)
        
        # Sum counts for each random sv
        all_shot_data = [sum_counts([all_shot_data[j][i] for j in range(len(all_shot_data))]) for i in range(len(self.rand_svs))]
        return all_shot_data

    async def run(self, circ: Circuit, data: PassData) -> None:
        self.final_svs = [circ.get_statevector(
            rand_sv) for rand_sv in self.rand_svs]
        
        self.true_counts = [get_shots(sv.get_probs(), 
                                      NUM_TOTAL_SHOTS) for sv in self.final_svs]
    

        un_futs = {}
        for ens_size in self.ens_sizes:
            print("Running TVD calculations", flush=True)
            output_file = f"ensemble_tvd_convergences_new/{self.circ_name}_{ens_size}_{self.tol}.pkl"
            if os.path.exists(output_file):
                continue
            print(f"Calculating Data for ensemble size {ens_size}", flush=True)
            # Calculate counts summed over all random initial SVs
            avg_uns_fut = get_runtime().map(self.get_full_counts_outer, 
                                            [ens_size] * self.num_trials)
            un_futs[ens_size] = avg_uns_fut

        for ens_size in un_futs:
            sampler_start = time.time()
            avg_uns = await un_futs[ens_size]
            all_counts: list[list[dict[str, int]]] = avg_uns
            # Calculate TVD for each trial
            all_data = []
            for counts in all_counts:
                # Get tvds for each random sv
                tvds = [tvd_dict(count, 
                                 self.true_counts[i]) for i, count in enumerate(counts)]
                # Pick max tvd
                all_data.append(np.max(tvds))
                
            print("All Data:", all_data, flush=True)
            td_time = time.time() - sampler_start
            print(f"Calculated data for ensemble size {ens_size} in {td_time:.2f} seconds", flush=True)
            output_file = f"ensemble_tvd_convergences_new/{self.circ_name}_{ens_size}_{self.tol}.pkl"
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'wb') as f:
                pickle.dump(all_data, f)
            print(f"Saved data to {output_file}", flush=True)

if __name__ == "__main__":
    circ_name = argv[1]
    tol = float(argv[2])
    small_ens = bool(int(argv[3])) if len(argv) > 3 else False
    compiler = Compiler(num_workers=250)
    cliff_t = False

    all_partitioned_data = pickle.load(open(partitioned_data_file, "rb"))

    total_circs_queried = 0
    
    if small_ens:
        ens_sizes = [1, 10, 50, 100, 500, 1000, 2000]
    else:
        ens_sizes = [4000, 8000, 16000, 64000]

    # for circ_name in circ_names:
    i = 0
    full_circ = load_circuit(circ_name)
    full_circ.remove_all_measurements()

    checkpoint_folder_form = f"{base_checkpoint_dir}/{circ_name}" + "_{large_block_num}_" + f"{tol}/"
    partitioned_circ_file=f"partitioned_circs/{circ_name}.pickle"
    partitioned_circ = pickle.load(open(partitioned_circ_file, "rb"))

    ens_pass = FullCircTDPass(
        circ_name=circ_name,
        tol=tol,
        num_qudits=full_circ.num_qudits,
        ens_sizes=ens_sizes,
        partitioned_circ=partitioned_circ,
        checkpoint_folder_form=checkpoint_folder_form,
        all_partitioned_data=all_partitioned_data,
        cliff_t=cliff_t
    )
    
    compiler.compile(full_circ, [ens_pass])
    compiler.close()

    