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

from .distance import trace_distance, get_density_matrix
from .dm_runner import generate_full_runner, DensityMatrixRunner
from .experiment_util import get_good_blocks

NUM_SAMPLES = 10


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

class DMEvaluator(BasePass):

    def __init__(self, circ_name: str, max_tol: float,
                 partitioned_data: dict[str, tuple[dict[str, Circuit], Circuit]],
                 ham: np.ndarray | None = None,
                 checkpoint_form: str = "",
                 partitioned_circ_file: str = "",
                 save_dir = "",
                 cliff_t: bool = False,
                 init_sv: StateVector = None,
                 run_td_also: bool = False) -> None:
        self.circ_name = circ_name
        self.ham = ham
        self.max_tol = max_tol
        self.save_dir = save_dir
        self.init_sv = init_sv
        self.checkpoint_form = checkpoint_form
        self.partitioned_circ_file = partitioned_circ_file
        self.cliff_t = cliff_t
        self.good_blocks, self.num_good_blocks = get_good_blocks(
            circ_name, 
            self.max_tol,
            self.cliff_t,
            checkpoint_form,
            partitioned_data
        )
        self.partitioned_data = update_partitioned_data(
            partitioned_data, self.good_blocks
        )

        self.block_runners: dict[tuple, tuple[DensityMatrixRunner, int]] = {}
        # Initialize a full circuit runner
        if self.num_good_blocks > 0:
            print("Good Blocks: ", self.good_blocks)
            print(list(partitioned_data.keys()), flush=True)
            self.full_circ_runner = generate_full_runner(
                circ_name=self.circ_name,
                max_tol=self.max_tol,
                partitioned_data=self.partitioned_data,
                checkpoint_form=self.checkpoint_form,
                partitioned_circ_file=self.partitioned_circ_file,
                cliff_t=self.cliff_t
            )
        self.run_td_also = run_td_also


    async def initialize(self) -> None:
        save_file = os.path.join(self.save_dir, f"{self.max_tol}_superop")
        await self.full_circ_runner.initialize(save_file)
        self.full_circ_runner.save(save_file)

    async def run_full_ensemble(self, svs: list[StateVector]) -> list[np.ndarray]:
        if len(svs) == 1:
            ens_data_file = os.path.join(self.save_dir, f"{self.max_tol}_rho_out.pkl")
        else:
            ens_data_file = os.path.join(self.save_dir, f"{self.max_tol}_rho_outs.pkl")
        if os.path.exists(ens_data_file):
            print(f"Ensemble data file {ens_data_file} already exists, skipping.", flush=True)
            return

        num_qubits = svs[0].num_qudits
        rho_ins = [get_density_matrix(sv.numpy) for sv in svs]
        rho_outs = [self.full_circ_runner.run(rho_in, np.arange(num_qubits)) for rho_in in rho_ins]
        return rho_outs

    async def run(self, circ: Circuit, data: PassData) -> None:
        if self.num_good_blocks == 0:
            print(f"No good blocks found for {self.circ_name} at tol {self.max_tol}, skipping.", flush=True)
            return

        await self.initialize()

        if self.ham is not None:
            print(f"Hamiltonian shape: {self.ham.shape}", flush=True)
            rho_outs = await self.run_full_ensemble([self.init_sv])
            if rho_outs is not None:
                rho = rho_outs[0]
                final_data = [(rho, self.init_sv)]
                file_name = os.path.join(self.save_dir, f"{self.max_tol}_rho_out.pkl")
                Path(file_name).parent.mkdir(parents=True, exist_ok=True)
                pickle.dump(final_data, open(file_name, "wb"))

        if (self.ham is None) or self.run_td_also:
            # Otherwise, run full circuit with ensemble
            rand_svs = [StateVector.random(circ.num_qudits) for _ in range(NUM_SAMPLES)]
            final_rho_outs = await self.run_full_ensemble(rand_svs)
            if final_rho_outs is None:
                return
            final_data = list(zip(final_rho_outs, rand_svs))
            print(f"Final data length: {len(final_data)}", flush=True)
            file_name = os.path.join(self.save_dir, f"{self.max_tol}_rho_outs.pkl")
            Path(file_name).parent.mkdir(parents=True, exist_ok=True)
            pickle.dump(final_data, open(file_name, "wb"))
