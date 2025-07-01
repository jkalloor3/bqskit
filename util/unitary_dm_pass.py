from itertools import chain
import numpy as np
from bqskit.ir.gates import CNOTGate
from bqskit.ir import Circuit, CircuitPoint
from bqskit.ir.circuit import CircuitGate
from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.ir.gates import ConstantUnitaryGate
from bqskit.qis import UnitaryMatrix, StateVector
import pickle
import os
import glob
import csv
from pathlib import Path
from .counter import load_avg_ensemble_counts_full, GateCounter
from .common import (create_avg_utry, load_jiggled_ensemble, 
                     get_block_names, create_jiggled_unitaries)
from .distance import trace_distance, get_density_matrix, frobenius_cost

from bqskit.runtime import get_runtime

NUM_SAMPLES = 30


def get_obs(dm: np.ndarray, ham: np.ndarray) -> float:
    '''
    Get the observable for the output density matrix.
    '''
    return np.real(np.trace(ham @ dm))


def get_final_dm(un: UnitaryMatrix) -> np.ndarray:
    '''
    Get the final density matrix for a circuit.
    '''

    sv = StateVector.zero(un.num_qudits)
    sv.apply(un, list(range(un.num_qudits)))
    dm =  get_density_matrix(sv.numpy)
    return dm

def get_final_dms(un: UnitaryMatrix) -> list[np.ndarray]:
    '''
    Get the final density matrix for a circuit.
    '''
    np.random.seed(42)
    final_dms = []
    for _ in range(10):
        sv = StateVector.random(un.num_qudits)
        sv.apply(un, list(range(un.num_qudits)))
        dm =  get_density_matrix(sv.numpy)
        final_dms.append(dm)
    return final_dms

def trace_distances(dms1: list[np.ndarray], target_dms: list[np.ndarray]) -> list[float]:
    '''
    Get the trace distances between two lists of density matrices.
    '''
    return [trace_distance(dm, target_dm) for dm, target_dm in zip(dms1, target_dms)]


def get_sub_block_nums(base_dir: str) -> list[str]:
    sub_block_path = f"{base_dir}/block_*.data"
    sub_block_files = glob.glob(sub_block_path)
    sub_block_nums = set()
    for sub_block_file in sub_block_files:
        sub_block_num = Path(sub_block_file).name.split("_")[-1].split(".")[0]
        sub_block_nums.add(sub_block_num)
    sub_block_nums = sorted(list(sub_block_nums))
    return sub_block_nums

def get_sub_block_count(large_block_dir: str,
                        small_block_num: str,
                        tol: float,
                        cliff_t: bool = False) -> tuple[bool, int]:
    qasm_file, jiggle_file, _, cache_file, csv_file = get_file_names(large_block_dir,
                                                   small_block_num)
    # Read CSV file, if the ratio is < 20 then we can read counts
    # print(qasm_file, csv_file, flush=True)
    if not os.path.exists(csv_file):        # print(f"CSV file {csv_file} does not exist.", flush=True)
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
    max_ratio = min(20, (10 ** (tol) / 10))
    if min_ratio > max_ratio:
        print(f"Skipping {small_block_num} as ratio is too high: {min_ratio}", flush=True)
        return False, 0
    # Now we need to get the avg. number of CNOTs
    if not cliff_t:
        qasm_str = open(qasm_file, 'r').read()
        count = qasm_str.count("cx ")
        num_circs = qasm_str.count("BREAK") + 1
        count = count / num_circs
    else:
        count = load_avg_ensemble_counts_full(qasm_file, jiggle_file, 
                                              cache_file, target_error=(10 ** (-tol)),
                                              count_t=True)

    return True, count


def get_file_names(large_checkpoint_dir, 
                   small_block_num: str) -> tuple[str, str, str, str, str]:
    checkpoint_dir = os.path.join(large_checkpoint_dir, f"block_{small_block_num}")
    ensemble_file_name = os.path.join(checkpoint_dir, "ensemble_{ind}_{extra}.qasms")
    jiggle_file_name = os.path.join(checkpoint_dir, "ensemble_{ind}_jiggles_{extra}.npy")
    cache_file_name = os.path.join(checkpoint_dir, "ensemble_{ind}_cache_{extra}.pkl")
    extra_str = "_fw"
    csv_file = os.path.join(large_checkpoint_dir, 
                                 f"block_{small_block_num}{extra_str}_fw.csv")
    if not os.path.exists(csv_file):
        csv_file = os.path.join(large_checkpoint_dir, 
                                 f"block_{small_block_num}.csv")
    ind = 0
    ensemble_file = ensemble_file_name.format(ind=ind, extra=extra_str)
    jiggle_file = jiggle_file_name.format(ind=ind, extra=extra_str)
    probs_file = f"{checkpoint_dir}/ensemble_final_probs_{extra_str}.npy"
    cache_file = cache_file_name.format(ind=ind, extra=extra_str)
    return ensemble_file, jiggle_file, probs_file, cache_file, csv_file


class UnitaryDMEvaluator(BasePass):

    def __init__(self, circ_name: str, max_tol: float,
                 partitioned_data: dict[str, tuple[dict[str, Circuit], Circuit]],
                 ham: np.ndarray | None = None,
                 checkpoint_form: str = "",
                 partitioned_circ_file: str = "",
                 save_dir = "",
                 cliff_t: bool = False) -> None:
        self.max_tol = max_tol
        self.circ_name = circ_name
        self.partitioned_data = partitioned_data
        self.ham = ham
        self.partitioned_circ_file = partitioned_circ_file
        self.save_dir = save_dir
        self.checkpoint_form = checkpoint_form
        self.cliff_t = cliff_t
        self.num_good_blocks = self.calculate_good_blocks()

    @staticmethod
    def generate_circ_unitary(large_block_gates: dict[str, ConstantUnitaryGate],
                                partitioned_circ: Circuit) -> UnitaryMatrix:
        num_digits = len(str(partitioned_circ.num_operations))
        circ = partitioned_circ.copy()
        for block_ind, (cycle, op) in enumerate(circ.operations_with_cycles()):
            pt = CircuitPoint(cycle, op.location[0])
            block_name = str(block_ind).zfill(num_digits)
            assert isinstance(op.gate, CircuitGate)
            assert isinstance(op.gate._circuit, Circuit)
            if block_name in large_block_gates:
                new_block_gate = large_block_gates[block_name]
                assert isinstance(new_block_gate, ConstantUnitaryGate)
                if (op.gate._circuit.num_qudits != new_block_gate.num_qudits):
                    print("Mismatch in qudits: ", block_name, 
                            op.gate._circuit.num_qudits, 
                            new_block_gate.num_qudits)
                assert op.gate._circuit.num_qudits == new_block_gate.num_qudits
                circ.replace_gate(pt, new_block_gate, op.location)
        return circ.get_unitary()

    @staticmethod
    async def generate_large_block_unitary(
                           block_files: dict[str, list[str]],
                           block_names: list[str],
                           block_targets: dict[str, UnitaryMatrix],
                           pcirc: Circuit) -> tuple[ConstantUnitaryGate, 
                                                    list[ConstantUnitaryGate]]:
        
        '''
        Returns a tuple of the average unitary gate for the block and a list of
        ConstantUnitaryGates for each sub-block.

        The list contains sample unitaries for each sub-block, in order to 
        calculate the average distance.
        
        '''
        block_gates = {}

        for block_name, files in block_files.items():
            circ_params = load_jiggled_ensemble(*files)
            target = block_targets[block_name]
            if len(circ_params) < 4:
                # Make 10 copies of each circuit and split the params amongst them
                new_circ_params = []
                for circ, params, probs, cache in circ_params:
                    param_chunks = np.array_split(params, 10)
                    probs_chunks = np.array_split(probs, 10)
                    for i in range(10):
                        if param_chunks[i].shape[0] > 5000:
                            # pick a random subset of 5000
                            rand_inds = np.random.choice(param_chunks[i].shape[0], 5000, replace=False)
                            param_chunks[i] = param_chunks[i][rand_inds]
                            probs_chunks[i] = probs_chunks[i][rand_inds]
                        new_circ_params.append((circ, param_chunks[i], probs_chunks[i], cache))
                circ_params = new_circ_params
            
            print("Calculating average unitary for block: ", block_name, flush=True)
            utries = await get_runtime().map(create_avg_utry, circ_params, target=target,
                                                            add_cost=False)
            avg_utry = np.sum(utries, axis=0)
            avg_gate = ConstantUnitaryGate(avg_utry)
            print(f"Block: {block_name}, Avg. Utry: {avg_utry.shape}", flush=True)
            avg_utry_dist = frobenius_cost(avg_utry, target)
            print(f"Block: {block_name}, Avg. Dist: {avg_utry_dist}", flush=True)
            if avg_utry_dist < 1:
                block_gates[block_name] = avg_gate
                # all_utries = await get_runtime().map(create_jiggled_unitaries,
                #                                      circ_params, target=target,
                #                                      add_cost=False)
                # all_utries = list(chain.from_iterable(all_utries))
                # # Randomly sample NUM_SAMPLES unitaries
                # use_duplicate_inds = len(all_utries) < NUM_SAMPLES
                # rand_inds = np.random.choice(len(all_utries), NUM_SAMPLES, 
                #                              replace=use_duplicate_inds)
                # all_utries = [all_utries[i] for i in rand_inds]
                # all_gates = [ConstantUnitaryGate(u) for u in all_utries]
                # avg_dist = np.mean(
                #     [frobenius_cost(u, target.numpy) for u in all_utries]
                # )
                # scaling_factor = avg_utry_dist / (avg_dist ** 2)
                # print(f"Block: {block_name}, Avg. Dist: {avg_utry_dist}, Scaling Factor: {scaling_factor}", flush=True)
                # block_utries[block_name] = all_gates

        full_circs = []
        # Get randomly sampled unitaries for each block
        # for i in range(NUM_SAMPLES):
        #     ind = 0
        #     circ = pcirc.copy()
        #     for cycle, op in circ.operations_with_cycles():
        #         pt = CircuitPoint(cycle, op.location[0])
        #         block_name = block_names[ind]
        #         ind += 1
        #         assert isinstance(op.gate, CircuitGate)
        #         assert isinstance(op.gate._circuit, Circuit)
        #         if block_name in block_utries:
        #             un_gate = block_utries[block_name][i]
        #             circ.replace_gate(pt, un_gate, op.location)

        #     full_circs.append(ConstantUnitaryGate(circ.get_unitary()))

        # Get average gate
        circ = pcirc.copy()
        ind = 0
        for cycle, op in circ.operations_with_cycles():
            pt = CircuitPoint(cycle, op.location[0])
            block_name = block_names[ind]
            ind += 1
            assert isinstance(op.gate, CircuitGate)
            assert isinstance(op.gate._circuit, Circuit)
            if block_name in block_gates:
                un_gate = block_gates[block_name]
                circ.replace_gate(pt, un_gate, op.location)

        avg_gate = ConstantUnitaryGate(circ.get_unitary())
        return (avg_gate, full_circs)

    async def get_block_ensemble(self, large_block_num:str) -> tuple[ConstantUnitaryGate, 
                                                                     list[ConstantUnitaryGate]]:
        # print("Loading Block: ", large_block_num, flush=True)
        if len(self.good_block_nums[large_block_num]) == 0:
            print(f"No good blocks for {large_block_num}, skipping.", flush=True)
            return None, []
        print("Loading Block: ", large_block_num, flush=True)

        small_block_circs, small_partitioned_circ = self.partitioned_data[large_block_num]
        
        block_files = {}
        block_targets = {}
        for small_block_num, small_circ in small_block_circs.items():
            block_targets[small_block_num] = small_circ.get_unitary()

        large_block_dir = self.checkpoint_form.format(large_block_num=large_block_num)
        
        for small_block_num  in self.good_block_nums[large_block_num]:
            # print("Loading Circuits: ", small_block_num)
            qasms_file, params_file, probs_file, cache_file, _ = get_file_names(large_block_dir, small_block_num)
            block_files[small_block_num] = [qasms_file, params_file, probs_file, cache_file]

        small_block_nums = get_sub_block_nums(large_block_dir)

        return await UnitaryDMEvaluator.generate_large_block_unitary(
            block_files=block_files,
            block_names=small_block_nums,
            block_targets=block_targets,
            pcirc=small_partitioned_circ
        )

    def calculate_good_blocks(self) -> None:
        self.good_block_nums = {}
        counter = GateCounter(est=False, cache_file=None)
        max_ratio = min(20, (10 ** (self.max_tol) / 2))
        large_block_nums = get_block_names(self.circ_name, extra="_tket")
        # print("Large Block Names: ", large_block_nums, flush=True)
        num_good_blocks = 0
        for large_block_num in large_block_nums:
            self.good_block_nums[large_block_num] = set()
            small_block_circs, _ = self.partitioned_data[large_block_num]
            large_block_dir = self.checkpoint_form.format(large_block_num=large_block_num)
            # print(f"Large Block Dir: {large_block_dir}", flush=True)
            for small_block_num, small_circ in small_block_circs.items():
                good, count = get_sub_block_count(large_block_dir, 
                                                  small_block_num, self.max_tol,
                                                  self.cliff_t)
                if self.cliff_t:
                    original_count = counter.count_t(small_circ, target_error=(10 ** (- 2 * self.max_tol) * max_ratio))
                else:
                    # Count CNOTs in the circuit
                    original_count = counter.count_cx(small_circ)
                if not good:
                    continue
                elif count > original_count:
                    print(f"Skipping {small_block_num} as count is too high: {count} >= {original_count}", flush=True)
                    continue
                else:
                    # Use block
                    num_good_blocks += 1
                    print(f"Adding {small_block_num} to {large_block_num} with count: {count}", flush=True)
                    self.good_block_nums[large_block_num].add(small_block_num)
        # print(list(self.good_block_nums.keys()))
        return num_good_blocks

    async def run_full_ensemble(self) -> None:
        ens_data_file = os.path.join(self.save_dir, f"{self.max_tol}.pkl")
        print("Ensemble Data File: ", ens_data_file, flush=True)
        if os.path.exists(ens_data_file):
            print("Already exists: ", ens_data_file, flush=True)
            return
        
        large_block_nums = get_block_names(self.circ_name, extra="_tket")
        # print("Large Block Names: ", large_block_nums, flush=True)
        print("Calculating Block Ensembles", self.circ_name, flush=True)
        block_unitaries_samples = await get_runtime().map(self.get_block_ensemble, 
                                                  large_block_nums)
        
        block_unitaries = [b[0] for b in block_unitaries_samples]
        # block_samples = [b[1] for b in block_unitaries_samples]

        # large_block_samples = dict(zip(large_block_nums, block_samples))
        # Remove all empty block circs
        # large_block_samples = {k: v for k, v in large_block_samples.items() if len(v) > 0}

        # all_block_samples = []
        # for ind in range(NUM_SAMPLES):
        #     sample_dict = {}
        #     for large_block_num, block_samples in large_block_samples.items():
        #         sample_dict[large_block_num] = block_samples[ind]
        #     all_block_samples.append(sample_dict)


        large_block_uns = dict(zip(large_block_nums, block_unitaries))
        # Remove all empty block circs
        large_block_uns = {k: v for k, v in large_block_uns.items() if v is not None}
        partitioned_circ = pickle.load(open(self.partitioned_circ_file, "rb"))

        print(f"Generating full ensemble for {self.circ_name} with {len(large_block_uns)} blocks", flush=True)
        full_un = UnitaryDMEvaluator.generate_circ_unitary(
            large_block_uns,
            partitioned_circ=partitioned_circ
        )
        print(f"Generated full unitary for {self.circ_name} with shape {full_un.shape}", flush=True)
        # print(f"Generating example unitaries for {self.circ_name} with {len(all_block_samples)} samples", flush=True)
        # example_uns = await get_runtime().map(
        #     UnitaryDMEvaluator.generate_circ_unitary,
        #     all_block_samples,
        #     partitioned_circ=partitioned_circ
        # )
        # example_uns = [
        #     UnitaryDMEvaluator.generate_circ_unitary(
        #         large_block_gates=sample,
        #         partitioned_circ=partitioned_circ
        #     ) for sample in all_block_samples
        # ]

        # print(f"Generated {len(example_uns)} example unitaries for {self.circ_name}", flush=True)

        # example_dms = await get_runtime().map(
        #     get_final_dm, example_uns
        # )
        if self.ham is None:
            full_dms = get_final_dms(full_un)
            # example_dms = [get_final_dms(un) for un in example_uns]
            ensemble_mag = np.max(trace_distances(full_dms, self.target_dms))
            # example_mags = []
            # for i, dms in enumerate(example_dms):
            #     example_mags.append(np.max(trace_distances(dms, self.target_dms)))
            # example_mags = await get_runtime().map(
            #     trace_distances, example_dms,
            #     target_dms=self.target_dms
            # )
            # example_mags = [np.max(mags) for mags in example_mags]
            # example_mags = []
        else:
            # example_dms = [get_final_dm(un) for un in example_uns]
            dm = get_final_dm(full_un)
            ensemble_mag = get_obs(dm, self.ham)
            # example_mags = await get_runtime().map(
            #     get_obs, example_dms, 
            #     ham=self.ham
            # )
            # example_mags = [get_obs(dm, self.ham) for dm in example_dms]
            # example_mags = []

        print(f"Ensemble Values for full circ: {ensemble_mag, 10 ** (-1 * self.max_tol)}", flush=True)
        Path(ens_data_file).parent.mkdir(parents=True, exist_ok=True)
        pickle.dump((ensemble_mag, 10 ** (-1 * self.max_tol), self.num_good_blocks), open(ens_data_file, "wb"))


    async def run(self, circ: Circuit, data: PassData) -> None:
        np.set_printoptions(precision=2, threshold=np.inf, linewidth=np.inf)
        self.target_dm = get_final_dm(circ.get_unitary())
        self.target_dms = get_final_dms(circ.get_unitary())
        
        await self.run_full_ensemble()