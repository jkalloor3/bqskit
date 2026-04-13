from itertools import chain
from math import ceil
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

from util.gg import GridSynthGate

from .distance import trace_distance, get_density_matrix
from .dm_runner import generate_all_block_runners, get_noisy_rho, get_single_rho
from .experiment_util import get_good_blocks
from .gg import get_approx_t_str
from .counter import fix_angle_workflow

NUM_SAMPLES = 2

def get_gg_circ_params_cache(circ: Circuit, tol: float) -> tuple[Circuit, np.ndarray, dict]:
    '''
    Get the default parameters cache for a circuit and convert to an RZ 
    '''
    target_error = 10 ** (- 2 * tol)

    # Calculate error per GG
    if circ.num_params == 0:
        return circ, circ.params, {}

    error_per_gg = target_error / circ.num_params
    precision = ceil(-np.log10(error_per_gg))
    new_circ = circ.copy()

    cache = fix_angle_workflow(new_circ, precision=precision, convert_to_gg=True)
    
    print("New Circ Gate Counts: ", new_circ.gate_counts)

    return new_circ, new_circ.params, cache


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

class NoisyDMEvaluator(BasePass):

    def __init__(self, circ_name: str, max_tol: float,
                 partitioned_data: dict[str, tuple[dict[str, Circuit], Circuit]],
                 checkpoint_form: str = "",
                 save_dir = "",
                 cliff_t: bool = False,
                 noise_level: float = 0) -> None:
        self.circ_name = circ_name
        self.max_tol = max_tol
        self.save_dir = save_dir
        self.checkpoint_form = checkpoint_form
        self.cliff_t = cliff_t
        self.noise_level = noise_level
        self.good_blocks, self.num_good_blocks = get_good_blocks(
            circ_name, 
            self.max_tol,
            self.cliff_t,
            checkpoint_form,
            partitioned_data
        )
        print("Good Blocks: ", self.good_blocks)
        self.partitioned_data = update_partitioned_data(
            partitioned_data, self.good_blocks
        )
        # Initialize block runners
        self.block_runners = generate_all_block_runners(
            circ_name=self.circ_name,
            max_tol=self.max_tol,
            partitioned_data=self.partitioned_data,
            checkpoint_form=self.checkpoint_form,
            cliff_t=self.cliff_t
        )

        # print("Block Runners: ", list(self.block_runners.keys()))
        # Only keep first 10 runners
        self.block_runners = {k: self.block_runners[k] for k in list(self.block_runners.keys())[:10]}

        print("Block Runners: ", list(self.block_runners.keys()))

        # Get original unitaries for good blocks
        self.block_circs: dict[tuple[str, str], UnitaryMatrix] = {}
        for block_num in self.block_runners:
            circ = partitioned_data[block_num[0]][0][block_num[1]]
            self.block_circs[block_num] = circ


    async def run_block_ensemble(self, 
                                 block_num: tuple[str, str],
                                 svs: list[StateVector]
                                 ) -> list[np.ndarray]:
        all_rho_outs = []
        block_runner = self.block_runners[block_num]

        await block_runner.initialize()
        
        for sv in svs:
            rho_out = block_runner.run_noisy(get_density_matrix(sv.numpy), self.noise_level)
            all_rho_outs.append(rho_out)
        
        return all_rho_outs
    

    async def run_block_exact(self,
                              block_circ: Circuit,
                                svs: list[StateVector]) -> list[np.ndarray]:
        

        new_circ, params, cache = get_gg_circ_params_cache(block_circ, self.max_tol)

        all_rho_outs = []
        for sv in svs:
            rho_out = get_noisy_rho(
                new_circ,
                params,
                cache,
                get_density_matrix(sv.numpy),
                self.noise_level
            )
            all_rho_outs.append(rho_out)

        return all_rho_outs


    async def run(self, circ: Circuit, data: PassData) -> None:
        if self.num_good_blocks == 0:
            print(f"No good blocks found for {self.circ_name} at tol {self.max_tol}, skipping.", flush=True)
            return

        # Otherwise, run full circuit with ensemble
        rho_futs = {}
        for block_num, block_circ in self.block_circs.items():
            # Save results
            file_name = os.path.join(self.save_dir, block_num[0],
                                    str(self.max_tol),
                                     str(self.noise_level),
                                     f"{block_num[1]}_{NUM_SAMPLES}.pkl")
            
            if os.path.exists(file_name):
                # print(f"File {file_name} already exists, skipping.", flush=True)
                continue

            rand_svs = [StateVector.random(block_circ.num_qudits) for _ in range(NUM_SAMPLES)]
            final_rho_outs_fut = self.run_block_ensemble(block_num, rand_svs)
            rho_futs[block_num] = (final_rho_outs_fut, rand_svs)

        for block_num, (final_rho_outs_fut, rand_svs) in rho_futs.items():
            block_circ = self.block_circs[block_num]
            block_un = block_circ.get_unitary()
            true_rho_outs = [get_single_rho(block_un, get_density_matrix(sv.numpy), tuple(range(block_un.num_qudits))) for sv in rand_svs]

            final_rho_outs = await final_rho_outs_fut
            tds = [trace_distance(rho_out, true_rho) for rho_out, true_rho in zip(final_rho_outs, true_rho_outs)]
        
            print(f"Final Trace Distances for block {block_num} at tol {self.max_tol}: {tds}", flush=True)

            # Save results
            file_name = os.path.join(self.save_dir, block_num[0],
                                    str(self.max_tol),
                                     str(self.noise_level),
                                     f"{block_num[1]}_{NUM_SAMPLES}.pkl")
            
            Path(file_name).parent.mkdir(parents=True, exist_ok=True)
            pickle.dump({
                'final_rho_outs': final_rho_outs,
                'true_rho_outs': true_rho_outs,
                "sv_ins": rand_svs,
                'tds': tds
            }, open(file_name, "wb"))


        # Run exact circuit for comparison
        for block_num, block_circ in self.block_circs.items():
            # Save results
            file_name = os.path.join(self.save_dir, block_num[0],
                                    str(self.max_tol),
                                     str(self.noise_level),
                                     f"{block_num[1]}_{NUM_SAMPLES}_exact_circuit.pkl")
            
            # if os.path.exists(file_name):
            #     print(f"Exact File {file_name} already exists, skipping.", flush=True)
            #     continue
            # else:
            print(f"Running exact circuit for block {block_num} at tol {self.max_tol}", flush=True)

            rand_svs = [StateVector.random(block_circ.num_qudits) for _ in range(NUM_SAMPLES)]

            rho_outs = await self.run_block_exact(block_circ, rand_svs)
            block_un = block_circ.get_unitary()
            true_rho_outs = [get_single_rho(block_un, get_density_matrix(sv.numpy), tuple(range(block_un.num_qudits))) for sv in rand_svs]

            tds = [trace_distance(rho_out, true_rho) for rho_out, true_rho in zip(rho_outs, true_rho_outs)]

            print(f"Exact Circuit Trace Distances for block {block_num} at tol {self.max_tol}: {tds}", flush=True)

            pickle.dump({
                'final_rho_outs': rho_outs,
                'true_rho_outs': true_rho_outs,
                "sv_ins": rand_svs,
                'tds': tds
            }, open(file_name, "wb"))
