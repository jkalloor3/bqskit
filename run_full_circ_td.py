from bqskit.ir.circuit import Circuit, CircuitGate
from bqskit.qis import StateVector, UnitaryMatrix
import time
import numpy as np
from pathlib import Path
import pickle
from sys import argv
import os
from typing import Generator

from bqskit.compiler import Compiler


from bqskit.qis.unitary.unitarybuilder import UnitaryBuilder
from util import  (get_file_names, load_jiggled_ensemble, EnsembleSampler, 
                   get_sub_block_count, GateCounter, load_circuit, trace_distance,
                   get_density_matrix, get_block_names, frobenius_cost)

from bqskit.runtime import get_runtime


from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData

cliff_t = False

NUM_RANDOM_SEEDS = 10

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

def get_final_dm(circs: list[Circuit], sv: StateVector) -> np.ndarray:
    '''
    For each random initialization, calculate the average final density matrix
    across all the circuits passed.
    '''
    avg_dm = np.zeros((2 ** circs[0].num_qudits, 2 ** circs[0].num_qudits), dtype=np.complex128)
    for circ in circs:
        sv_out = circ.get_statevector(sv)
        dm =  get_density_matrix(sv_out.numpy)
        avg_dm += dm
    avg_dm /= len(circs)
    return avg_dm

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
    
    saved_utries = {}

    while True:
        # block_circ = p_circ.copy()
        # print(good_blocks, flush=True)
        block_un = UnitaryBuilder(p_circ.num_qudits, p_circ.radixes)
        small_num_digits = len(str(p_circ.num_operations))
        i = 0
        for cycle, op in p_circ.operations_with_cycles():
            assert isinstance(op.gate, CircuitGate), "Operation is not a CircuitGate"
            small_block_num = str(i).zfill(small_num_digits)
            i += 1
            if small_block_num not in good_blocks:
                if small_block_num in saved_utries:
                    small_utry = saved_utries[small_block_num]
                else:
                    small_utry = op.get_unitary()
                    saved_utries[small_block_num] = small_utry
                saved_utries[small_block_num] = small_utry
                block_un.apply_right(small_utry, op.location)
                continue
            # Sample a small block circuit
            # print("Sampling small block:", small_block_num, flush=True)
            small_circ = next(circ_samplers[small_block_num])
            block_un.apply_right(small_circ.get_unitary(), op.location)
        yield block_un.get_unitary()

def generate_full_circ(circ_name: str, 
                       big_partitioned_circ: Circuit,
                       tol: float,
                       good_blocks: dict[str, set[str]],
                       partitioned_data: dict[str, Circuit]) -> Generator[np.ndarray, 
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
    orig_unitaries = {}
    i = 0
    large_block_num_digits = len(str(big_partitioned_circ.num_operations))
    for cycle, op in big_partitioned_circ.operations_with_cycles():
        assert isinstance(op.gate, CircuitGate), "Operation is not a CircuitGate"
        large_block_num = str(i).zfill(large_block_num_digits)
        i += 1
        if large_block_num not in good_blocks:
            orig_unitaries[large_block_num] = op.get_unitary()
            continue
        large_circ_generators[large_block_num] = generate_large_block_circ(
            circ_name,
            large_block_num,
            tol,
            partitioned_data[large_block_num][1],
            good_blocks[large_block_num]
        )

    while True:
        # out_circ = big_partitioned_circ.copy()
        utry = UnitaryBuilder(big_partitioned_circ.num_qudits, big_partitioned_circ.radixes)
        i = 0
        for op in big_partitioned_circ.operations():
            assert isinstance(op.gate, CircuitGate), "Operation is not a CircuitGate"
            large_block_num = str(i).zfill(large_block_num_digits)
            i += 1
            if large_block_num not in good_blocks:
                utry.apply_right(orig_unitaries[large_block_num], op.location)
            else:
                sample_un = next(large_circ_generators[large_block_num])
                utry.apply_right(sample_un, op.location)

        yield utry.get_unitary().numpy

def get_ens_td(un: np.ndarray, rand_sv: np.ndarray, true_dm: np.ndarray) -> float:
    '''
    Get the trace distance for an ensemble of circuits.
    '''
    sv_out = un @ rand_sv
    ens_dm = get_density_matrix(sv_out)
    td = trace_distance(ens_dm, true_dm)
    return td

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
                 cliff_t: bool = False,
                 run_td: bool = True) -> None:
        super().__init__()
        self.circ_name = circ_name
        self.tol = tol
        self.ens_sizes = ens_sizes
        self.num_qudits = num_qudits
        self.rand_svs = [StateVector.random(num_qudits) for _ in range(NUM_RANDOM_SEEDS)]
        # self.full_circ_generator = generate_full_circ(
        #     circ_name=circ_name,
        #     big_partitioned_circ=partitioned_circ,
        #     tol=tol,
        #     good_blocks=get_good_blocks(circ_name=circ_name,
        #                                 tol=tol,
        #                                 cliff_t=cliff_t,
        #                                 checkpoint_folder_form=checkpoint_folder_form,
        #                                 partitioned_data=all_partitioned_data[circ_name]),
        #     partitioned_data=all_partitioned_data[circ_name]
        # )
        self.partitioned_circ = partitioned_circ
        self.good_blocks = get_good_blocks(circ_name=circ_name,
                                           tol=tol,
                                           cliff_t=cliff_t,
                                           checkpoint_folder_form=checkpoint_folder_form,
                                           partitioned_data=all_partitioned_data[circ_name])
        self.partitioned_data = all_partitioned_data[circ_name]
        self.num_trials = 6
        self.run_td = run_td


    async def get_trial_un(self, ens_size: int) -> np.ndarray:
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
        avg_un = np.zeros((2 ** self.num_qudits, 2 ** self.num_qudits), dtype=np.complex128)
        for _ in range(ens_size):
            avg_un += next(full_circ_generator)
        avg_un /= ens_size
        return avg_un
    
    async def get_trial_un_outer(self, ens_size: int) -> np.ndarray:
        '''
        Get the average unitary for a given ensemble size.
        '''

        # Split ens_size into chunks of 2000
        CHUNK_SIZE = 2000
        if ens_size > CHUNK_SIZE:
            split_ens = [CHUNK_SIZE] * (ens_size // CHUNK_SIZE)
        else:
            split_ens = [ens_size]
        avg_uns = await get_runtime().map(self.get_trial_un, split_ens)

        # Average the unitaries
        avg_un = np.mean(avg_uns, axis=0)
        return avg_un

    def get_trial_td(self, rand_sv, dm, un) -> float:
        '''
        Get the trace distance for a given average unitary.
        '''
        sv_out = un @ rand_sv
        ens_dm = get_density_matrix(sv_out)
        td = trace_distance(ens_dm, dm)
        return td

    async def run(self, circ: Circuit, data: PassData) -> None:
        # print("Running FullCircTDPass", flush=True)
        self.true_dms = [get_final_dm([circ], sv=rand_sv) for rand_sv in self.rand_svs]
        self.full_un = circ.get_unitary()

        un_futs = {}
        for ens_size in self.ens_sizes:
            if self.run_td:
                output_file = f"ensemble_td_convergences_new/{self.circ_name}_{ens_size}_{self.tol}.pkl"
            else:
                output_file = f"ensemble_bias_convergences_new/{self.circ_name}_{ens_size}_{self.tol}.pkl"
            if os.path.exists(output_file):
                continue
            print(f"Calculating Data for ensemble size {ens_size}", flush=True)
            avg_uns_fut = get_runtime().map(self.get_trial_un_outer, [ens_size] * self.num_trials)
            un_futs[ens_size] = avg_uns_fut

        for ens_size in un_futs:
            sampler_start = time.time()
            avg_uns = await un_futs[ens_size]
            sample_time = time.time() - sampler_start
            print(f"Sampled average unitaries for ensemble size {ens_size} in {sample_time:.2f} seconds", flush=True)
            if self.run_td:
                all_data = []
                for i, un in enumerate(avg_uns):
                    rand_tds = []
                    for j, rand_sv in enumerate(self.rand_svs):
                        td = self.get_trial_td(rand_sv, self.true_dms[j], un)
                        rand_tds.append(td)
                    print("Max TD:", np.max(rand_tds), flush=True)
                    all_data.append(np.max(rand_tds))
            else:
                # Calculate bias for each un
                all_data = [frobenius_cost(un, self.full_un) for un in avg_uns]

            print("All Data:", all_data, flush=True)
            
            td_time = time.time() - sampler_start
            print(f"Calculated data for ensemble size {ens_size} in {td_time:.2f} seconds", flush=True)
            if self.run_td:
                output_file = f"ensemble_td_convergences_new/{self.circ_name}_{ens_size}_{self.tol}.pkl"
            else:
                output_file = f"ensemble_bias_convergences_new/{self.circ_name}_{ens_size}_{self.tol}.pkl"
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'wb') as f:
                pickle.dump(all_data, f)
            print(f"Saved data to {output_file}", flush=True)

if __name__ == "__main__":
    circ_name = argv[1]
    tol = float(argv[2])
    small_ens = bool(int(argv[3])) if len(argv) > 3 else False
    run_td = bool(int(argv[4])) if len(argv) > 4 else True
    compiler = Compiler(num_workers=128)
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
        cliff_t=cliff_t,
        run_td=run_td
    )
    
    compiler.compile(full_circ, [ens_pass])
    compiler.close()

    