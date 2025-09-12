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
from .dm_runner import generate_full_runner
from .distance import trace_distance, get_density_matrix
from .common import get_block_names

from bqskit.runtime import get_runtime

NUM_SAMPLES = 6


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

def get_final_dms(un: UnitaryMatrix, rand_svs: list[StateVector]) -> list[np.ndarray]:
    '''
    Get the final density matrix for a circuit.
    '''
    final_dms = []
    for rand_sv in rand_svs:
        sv = StateVector(rand_sv.numpy)
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

    # Now we need to check if the ratio is less than 20
    max_ratio = min(20, (10 ** (tol) / 4))

    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        min_ratio = float("inf")
        final_frob_cost = float("inf")
        for row in reader:
            if "Ratio" in row:  # Check if the column value is not empty
                # Check if distance is less than 3* 10^(-tol)
                max_dist = max_ratio * (10 ** (-tol))
                dist = float(row["Epsilon"])
                if dist > max_dist:
                    continue
                min_ratio = min(min_ratio, float(row["Ratio"]))
                final_frob_cost = min(final_frob_cost, float(row["Norm. Bias"]))


    if min_ratio > max_ratio:
        # print(f"Skipping {small_block_num} as ratio is too high: {min_ratio}", flush=True)
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

def calculate_good_blocks(circ_name: str, 
                          checkpoint_form: str, 
                          max_tol: float, 
                          cliff_t: bool) -> tuple[dict[str, set[str]], int]:
    large_block_nums = get_block_names(circ_name, extra="_tket")
    print("Large Block Nums: ", large_block_nums)
    good_blocks: dict[str, set[str]] = {}
    num_good_blocks = 0
    for large_block_num in large_block_nums:
        good_blocks[large_block_num] = set()
        large_block_dir = checkpoint_form.format(circ_name=circ_name,
                                                  large_block_num=large_block_num,
                                                  max_tol=max_tol)
        small_block_nums = get_sub_block_nums(large_block_dir)
        for small_block_num in small_block_nums:
            good, _ = get_sub_block_count(large_block_dir,
                                                small_block_num, max_tol,
                                                cliff_t)
            if not good:
                continue
            else:
                good_blocks[large_block_num].add(small_block_num)
                num_good_blocks += 1
    return good_blocks, num_good_blocks

def update_partitioned_data(partitioned_data: dict[str, tuple[dict[str, Circuit], Circuit]],
                            good_blocks: dict[str, set[str]]) -> dict[str, tuple[dict[str, Circuit], Circuit]]: 
    new_partitioned_data = {}
    for large_block_num in good_blocks:
        sub_partitioned_data, p_circ = partitioned_data[large_block_num]
        new_sub_partitioned_data = {}
        for small_block_num in good_blocks[large_block_num]:
            new_sub_partitioned_data[small_block_num] = sub_partitioned_data[small_block_num]
        new_partitioned_data[large_block_num] = (new_sub_partitioned_data, p_circ)
    return new_partitioned_data


def get_file_names(large_checkpoint_dir, 
                   small_block_num: str,
                   no_qp: bool = False) -> tuple[str, str, str, str, str]:
    small_checkpoint_dir = os.path.join(large_checkpoint_dir, f"block_{small_block_num}")

    if no_qp:
        ensemble_file = os.path.join(small_checkpoint_dir, "ensemble_final_fw_no_qp.qasms")
        jiggle_file = os.path.join(small_checkpoint_dir, "ensemble_final_jiggle_fw_no_qp.npy")
        cache_file = os.path.join(small_checkpoint_dir, "ensemble_final_cache_fw_no_qp.pkl")
        csv_file = os.path.join(large_checkpoint_dir, f"block_{small_block_num}_fw_no_qp.csv")
        return ensemble_file, jiggle_file, final_probs_file, cache_file, csv_file

    # Try outputs of newest passes
    final_probs_file = os.path.join(small_checkpoint_dir, "ensemble_final_probs_fw.npy")
    if os.path.exists(final_probs_file):
        ensemble_file = os.path.join(small_checkpoint_dir, "ensemble_final_fw.qasms")
        jiggle_file = os.path.join(small_checkpoint_dir, "ensemble_final_jiggle_fw.npy")
        cache_file = os.path.join(small_checkpoint_dir, "ensemble_final_cache_fw.pkl")
        csv_file = os.path.join(large_checkpoint_dir, f"block_{small_block_num}_fw.csv")
        return ensemble_file, jiggle_file, final_probs_file, cache_file, csv_file
    
    # Otherwise, we do not have the newest set of files, so return the old ones
    csv_file = os.path.join(large_checkpoint_dir,
                             f"block_{small_block_num}_fw.csv")
    if not os.path.exists(csv_file):
        csv_file = os.path.join(large_checkpoint_dir, 
                                 f"block_{small_block_num}.csv")
    ensemble_file = os.path.join(small_checkpoint_dir, "ensemble_0__fw.qasms")
    jiggle_file = os.path.join(small_checkpoint_dir, "ensemble_0_jiggles__fw.npy")
    probs_file = os.path.join(small_checkpoint_dir, "ensemble_0_probs__fw.npy")
    cache_file = os.path.join(small_checkpoint_dir, "ensemble_0_cache__fw.pkl")
    return ensemble_file, jiggle_file, probs_file, cache_file, csv_file


class DMEvaluator(BasePass):

    def __init__(self, circ_name: str, max_tol: float,
                 partitioned_data: dict[str, tuple[dict[str, Circuit], Circuit]],
                 ham: np.ndarray | None = None,
                 checkpoint_form: str = "",
                 partitioned_circ_file: str = "",
                 save_dir = "",
                 cliff_t: bool = False,
                 init_sv: StateVector = None) -> None:
        self.circ_name = circ_name
        self.ham = ham
        self.max_tol = max_tol
        self.save_dir = save_dir
        self.init_sv = init_sv
        self.checkpoint_form = checkpoint_form
        self.partitioned_circ_file = partitioned_circ_file
        self.cliff_t = cliff_t
        self.good_blocks, self.num_good_blocks = calculate_good_blocks(
            circ_name, checkpoint_form, max_tol, cliff_t
        )
        print("Good Blocks: ", self.good_blocks)
        self.partitioned_data = update_partitioned_data(
            partitioned_data, self.good_blocks
        )

        self.full_circ_runner = generate_full_runner(
            circ_name=self.circ_name,
            max_tol=self.max_tol,
            partitioned_data=self.partitioned_data,
            checkpoint_form=self.checkpoint_form,
            partitioned_circ_file=self.partitioned_circ_file,
            cliff_t=self.cliff_t
        )

    async def run_full_ensemble(self, sv: StateVector) -> None:
        ens_data_file = os.path.join(self.save_dir, f"{self.max_tol}.pkl")
        if os.path.exists(ens_data_file):
            print(f"Ensemble data file {ens_data_file} already exists, skipping.", flush=True)
            return
        
        rho_in = get_density_matrix(sv.numpy)
        await self.full_circ_runner.initialize()
        num_qubits = sv.num_qudits
        rho_out = self.full_circ_runner.run(rho_in, np.arange(num_qubits))

        return rho_out

    async def run(self, circ: Circuit, data: PassData) -> None:
        if self.num_good_blocks == 0:
            print(f"No good blocks found for {self.circ_name} at tol {self.max_tol}, skipping.", flush=True)
            return

        if self.ham is not None:
            # out_sv = circ.get_statevector(self.init_sv)
            # self.target_dm = get_density_matrix(out_sv.numpy)
            # self.obs = get_obs(self.target_dm, self.ham)
            # print(f"Target Observable: {self.obs}", flush=True)
            rho_out = await self.run_full_ensemble(self.init_sv)
            final_data = [(rho_out, self.init_sv)]
        else:
            rand_svs = [StateVector.random(circ.num_qudits) for _ in range(NUM_SAMPLES)]
            # target_dms = get_final_dms(circ.get_unitary(), rand_svs)
            # final_data = []
            # for rand_sv, target_dm in zip(rand_svs, target_dms):
            #     rho_out = await self.run_full_ensemble(rand_sv)
            #     final_data.append((rho_out, rand_sv, target_dm))
            final_rho_outs = await get_runtime().map(self.run_full_ensemble, rand_svs)
            final_data = list(zip(final_rho_outs, rand_svs))
        
        # Save output rho
        rho_file = os.path.join(self.save_dir, f"{self.max_tol}_rho_outs.pkl")
        Path(rho_file).parent.mkdir(parents=True, exist_ok=True)
        # np.save(rho_file, rho_out)
        pickle.dump(final_data, open(rho_file, "wb"))

        # ensemble_mag = get_obs(rho_out, self.ham) - self.obs

        # print(f"Ensemble Values for full circ: {ensemble_mag, 10 ** (-1 * self.max_tol)}", flush=True)
        # Path(ens_data_file).parent.mkdir(parents=True, exist_ok=True)
        # pickle.dump((ensemble_mag, 10 ** (-1 * self.max_tol), len(self.good_blocks)), open(ens_data_file, "wb"))
