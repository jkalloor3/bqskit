from bqskit.ir.circuit import Circuit, CircuitGate
from bqskit.qis import StateVector, UnitaryMatrix
import time
import numpy as np
from pathlib import Path
import pickle
from sys import argv
import os
import csv
from typing import Generator

from bqskit.compiler import Compiler

from util.experiment_util import get_good_blocks, get_file_names
from util.distance import trace_distance, get_density_matrix, operator_norm
from util.common import  load_jiggled_ensemble
from util.samplers import EnsembleUnitarySampler


from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData

from bqskit.qis.unitary.unitarybuilder import UnitaryBuilder
from bqskit.runtime import get_runtime

cliff_t = False

NUM_RANDOM_SEEDS = 10

if cliff_t:
    # print("Using cliff-t circuits", flush=True)
    base_checkpoint_dir = "small_block_checkpoints_final_paper_4_clifft_tket"
    partitioned_data_file = "partitioned_data_all_circs_clifft.pickle"
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

def get_good_block_sizes(circ_name, tol, cliff_t,
                    checkpoint_folder_form: str,
                    partitioned_data: dict) -> dict[tuple[str, str], int]:
    '''
    Get the good blocks for each circuit from the checkpoint folder.
    
    checkpoint_folder_form: The form of the checkpoint folder path.
    '''
    good_blocks, _ = get_good_blocks(circ_name=circ_name,
                                  tol=tol,
                                  cliff_t=cliff_t,
                                  checkpoint_folder_form=checkpoint_folder_form,
                                  partitioned_data=partitioned_data)
    good_blocks_sizes = {}
    for large_block_num in good_blocks:
        small_block_circs, _ = partitioned_data[large_block_num]
        for small_block_num in good_blocks[large_block_num]:
            small_circ: Circuit = small_block_circs[small_block_num]
            good_blocks_sizes[(large_block_num, small_block_num)] = small_circ.num_qudits
    return good_blocks_sizes


def get_ens_td(un: np.ndarray, rand_sv: np.ndarray, true_dm: np.ndarray) -> float:
    '''
    Get the trace distance for an ensemble of circuits.
    '''
    sv_out = un @ rand_sv
    ens_dm = get_density_matrix(sv_out)
    td = trace_distance(ens_dm, true_dm)
    return td

class FullCircvRPass(BasePass):
    '''
    A pass that runs the full circuit trace distance calculation.
    '''

    def __init__(self, 
                 circ_name: str, 
                 tol: float,
                 checkpoint_folder_form: str,
                 all_partitioned_data: dict[str, Circuit],
                 cliff_t: bool = False) -> None:
        super().__init__()
        self.circ_name = circ_name
        self.tol = tol
        self.good_blocks_sizes = get_good_block_sizes(circ_name=circ_name,
                                           tol=tol,
                                           cliff_t=cliff_t,
                                           checkpoint_folder_form=checkpoint_folder_form,
                                           partitioned_data=all_partitioned_data[circ_name])
        self.partitioned_data = all_partitioned_data[circ_name]
        self.num_trials = 10

    async def get_trial_vR(self, 
                           block_num: tuple[str, str]
                           ) -> tuple[list[float], list[float]]:
        '''
        Get the average unitary for a given ensemble size.
        '''
        large_block_num, small_block_num = block_num
        files = get_file_names(
            base_dir_form.format(circ_name=circ_name, 
                                 large_block_num=large_block_num, 
                                 tol=tol),
            small_block_num=small_block_num
        )[:-1]

        block_generator = EnsembleUnitarySampler(
            load_jiggled_ensemble(*files),
            cliff_t=cliff_t
        )

        rho_bar = self.rho_bars[block_num]
        sv_in = self.svs[block_num]

        rho_in = get_density_matrix(sv_in.numpy) 

        max_R = -1 * float("inf")
        # Options for V bc Idk what Mohan means here
        v_sum_1 = np.zeros_like(rho_in)
        v_sum_2 = np.zeros_like(rho_in)
        v_sum_3 = np.zeros_like(rho_in)
        for un, p_i in block_generator:
            rho_i = un @ rho_in @ un.conj().T
            rho_o = rho_i - rho_bar
            R = operator_norm(rho_o)
            max_R = max(max_R, R)
            v_sum_1 += p_i * (rho_o @ rho_o)
            v_sum_2 += p_i * (rho_o.conj().T @ rho_o)
            v_sum_3 += p_i * (rho_o ** 2)

        v_1 = operator_norm(v_sum_1)
        v_2 = operator_norm(v_sum_2)
        v_3 = operator_norm(v_sum_3)
        v = [v_1, v_2, v_3]

        return v, max_R


    async def get_rho_bar_block(self, 
                                block_num: tuple[str, str], 
                                sv: StateVector) -> np.ndarray:
        '''
        Get the average final density matrix for a given block and random initialization.
        '''
        large_block_num, small_block_num = block_num
        files = get_file_names(
            base_dir_form.format(circ_name=circ_name, 
                                 large_block_num=large_block_num, 
                                 tol=tol),
            small_block_num=small_block_num
        )[:-1]

        block_generator = EnsembleUnitarySampler(
            load_jiggled_ensemble(*files),
            cliff_t=cliff_t
        )
        return block_generator.get_rho_out(sv.numpy)


    async def run(self, circ: Circuit, data: PassData) -> None:

        output_file = f"ensemble_vRs_final_2/{self.circ_name}_{self.tol}.pkl"
        if os.path.exists(output_file):
            print(f"File {output_file} already exists. Skipping.", flush=True)
            return
        
        block_nums = list(self.good_blocks_sizes.keys())

        if len(block_nums) == 0:
            print(f"No good blocks found for {self.circ_name} with tol {self.tol}. Skipping.", flush=True)
            return

        rho_bars_fut = []
        svs = []

        for block_num in block_nums:
            sv = StateVector.random(self.good_blocks_sizes[block_num])
            svs.append(sv)
            rho_bars_fut.append(self.get_rho_bar_block(block_num, sv))

        rho_bars = [await fut for fut in rho_bars_fut]
        self.rho_bars = dict(zip(block_nums, rho_bars))

        self.svs = dict(zip(block_nums, svs))

        vRs = await get_runtime().map(self.get_trial_vR, block_nums)
        
        all_vRs = dict(zip(block_nums, vRs))

        Path(output_file).parent.mkdir(parents=True, exist_ok=True)

        pickle.dump((all_vRs, self.rho_bars, self.svs), 
                    open(output_file, "wb"))


if __name__ == "__main__":
    circ_names = ["heisenberg7", "qaoa10"]
    compiler = Compiler(num_workers=256)
    cliff_t = False
  
    all_partitioned_data = pickle.load(open(partitioned_data_file, "rb"))

    ids = []
    for circ_name in circ_names:
        partitioned_circ_file=f"partitioned_circs/{circ_name}.pickle"
        partitioned_circ = pickle.load(open(partitioned_circ_file, "rb"))

        tols = [1.0, 2.0, 3.0, 4.0, 5.0]

        for tol in tols:
            print(f"Running {circ_name} with tol {tol}", flush=True)
            checkpoint_folder_form = f"{base_checkpoint_dir}/{circ_name}" + "_{large_block_num}_" + f"{tol}/"
            ens_pass = FullCircvRPass(
                circ_name=circ_name,
                tol=tol,
                checkpoint_folder_form=checkpoint_folder_form,
                all_partitioned_data=all_partitioned_data,
                cliff_t=cliff_t
            )
            
            id = compiler.submit(circuit=Circuit(7), workflow=[ens_pass])
            ids.append(id)

    for id in ids:
        compiler.result(id)
        
    compiler.close()

    