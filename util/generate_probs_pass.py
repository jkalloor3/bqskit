"""This module implements the InstantiateCount pass"""
from __future__ import annotations

from typing import Any

from bqskit.ir import Circuit
from bqskit.qis import UnitaryMatrix
from bqskit.ir.gates import CNOTGate
from bqskit.runtime import get_runtime
from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
import numpy as np
from multiprocessing import shared_memory
from .common import load_jiggled_ensemble_separate, create_jiggled_unitaries_shm
from .distance import frobenius_cost, normalized_frob_cost
from qpsolvers import solve_ls
import pickle
import os

NUM_CIRCS_PER_PROB = 1000

async def generate_probs(circuits: list[Circuit], 
                              params: np.ndarray, 
                              target: UnitaryMatrix, 
                              shm_name: str) -> np.ndarray:
    # Check if circ_params are too large
    assert len(circuits) == params.shape[0]

    # Calculate in chunks of 20
    num_chunks = 20
    chunk_params = np.array_split(params, num_chunks, axis=1)

    existing_shm = shared_memory.SharedMemory(name=shm_name)
    existing_shm_ret = shared_memory.SharedMemory(name=shm_name + "_ret")

    print(f"Sending {len(chunk_params)} times", flush=True)
    all_probs = []
    for chunk_ind, circ_param_chunk in enumerate(chunk_params):
        # print("Chunk Size", shm_size, circ_param_chunk.shape, flush=True)
        shared_array = np.ndarray(circ_param_chunk.shape, dtype=np.float64, buffer=existing_shm.buf)
        shared_array[:] = circ_param_chunk[:]
        print("Wrote to Shared Memory", flush=True)
        param_inds = np.arange(circ_param_chunk.shape[0])
        circ_inds = list(zip(circuits, param_inds))
        # Create return array in shared memory
        num_circs_ret = circ_param_chunk.shape[0] * circ_param_chunk.shape[1]
        # Of Size (NUM_CIRCS_RET, dim, dim)
        dim = target.shape[0]
        shm_ret_shape = (num_circs_ret, dim, dim)
        print("Shared Memory Ret Shape", shm_ret_shape, flush=True)
        shared_array_ret = np.ndarray(shm_ret_shape, dtype=np.complex128, buffer=existing_shm_ret.buf)
        # Load with zeros
        shared_array_ret[:] = 0
        await get_runtime().map(create_jiggled_unitaries_shm , circ_inds, 
                                shm_name=shm_name, 
                                shm_ret_name=shm_name + "_ret",
                                shm_shape=circ_param_chunk.shape,
                                shm_ret_shape=shm_ret_shape,
                                target=target)
        print("Put Unitaries into Shared Buffer")
        probs = await GenerateProbabilityPass.calculate_probs(shm_name + "_ret", shm_ret_shape, target)
        print(f"Calculated Probs for {len(probs)} circs", flush=True)
        all_probs.append(probs)
    existing_shm.close()
    existing_shm_ret.close()
    return all_probs


class GenerateProbabilityPass(BasePass):
    
    def __init__(self, shm_percentage: float = 1.0) -> None:
        super().__init__()
        self.shm_percentage = shm_percentage

    @staticmethod
    async def calculate_probs(shm_name: str, shm_shape: tuple[int, int, int], target: np.ndarray) -> np.ndarray:
        existing_shm = shared_memory.SharedMemory(name=shm_name)
        ensemble = np.ndarray(shm_shape, dtype=np.complex128, buffer=existing_shm.buf)
        M = len(ensemble)

        print(ensemble.shape, flush=True)

        tr_V_Us = np.einsum("mij,ij->m", ensemble, target.conj())
        tr_Us = np.einsum("aij,bij->ab", ensemble.conj(), ensemble)

        sample_inds = np.random.choice(M, size=5, replace=True)
        print("Sample Dists", [tr_V_Us[i] for i in sample_inds], flush=True)
        print("Sample Covars", [tr_Us[i][0] for i in sample_inds], flush=True)

        print("Finished Calculating Ensemble Vectors", flush=True)
        print(tr_V_Us.shape, tr_Us.shape, flush=True)

        # Create f and H matrices
        f = -2 * np.real(tr_V_Us)
        H = 2 * np.real(tr_Us)

        # Make pos definite
        isposdef = False
        trials = 0
        while not isposdef and trials < 20:
            try:
                R = np.linalg.cholesky(H)
                isposdef = True
            except np.linalg.LinAlgError:
                # Off by a little
                H += 1e-10 * np.eye(M)
                print(f"Perturbing by a little to make pos def trial num: {trials}")
                isposdef = False
                trials += 1

        if not isposdef:
            print('H not positive definite by a lot! Returning uniform dist')
            return [1 / len(ensemble) for _ in ensemble]
        
        # Constraints, probabilities should sum to 1 and be between 0 and 1
        Aeq = np.ones((1, M))
        beq = np.array([1])
        lbound = np.zeros(M)
        ubound = np.ones(M)

        # Solve with LS since it is convex
        s = -1 * np.linalg.inv(R) @ f
        probabilities = solve_ls(R.T, s, None, None, Aeq, beq, lbound, ubound, solver='clarabel')

        sample_probs = np.random.choice(probabilities, size=10, replace=True)
        print("Sample Probs", sample_probs, flush=True)

        return probabilities

    async def run(
            self, 
            circuit : Circuit, 
            data: PassData
    ) -> None:

        print("Running Generate Probability Pass", flush=True)
        checkpoint_dir = data["checkpoint_dir"]
        final_ens_file = f"{checkpoint_dir}/ensemble_final.qasms"
        final_ens_jiggle_file = f"{checkpoint_dir}/ensemble_final_jiggle.npy"
        probs_file = f"{checkpoint_dir}/ensemble_final_probs.npy"


        if os.path.exists(probs_file):
            print("Already calculated probabilities, skipping", flush=True)
            return

        shm_name = checkpoint_dir.split("/")[-1]
        print("Shared Memory Name: ", shm_name, flush=True)

        circuits, params = load_jiggled_ensemble_separate(final_ens_file, final_ens_jiggle_file)
        all_probs = await generate_probs(circuits, params,target=data.target, shm_name=shm_name)

        # Concatenate all probs
        all_probs = np.concatenate(all_probs, axis=0)
        # Divide by 20
        all_probs /= 20

        print("Calculated all probabilities", flush=True)

        if "checkpoint_dir" in data:
            np.save(probs_file, all_probs)
        return

