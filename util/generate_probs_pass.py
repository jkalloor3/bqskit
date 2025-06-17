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
from .common import load_jiggled_ensemble, create_jiggled_unitaries
from .distance import frobenius_cost, normalized_frob_cost
from qpsolvers import solve_qp
import pickle
import os

NUM_CIRCS_PER_PROB = 5000

class GenerateProbabilityPass(BasePass):
    
    def __init__(self, run_on_ensemble_0: bool = False) -> None:
        self.run_on_ensemble_0 = run_on_ensemble_0



    @staticmethod
    def calculate_probs(ensemble: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Calculate the probabilities for the ensemble"""
        M = ensemble.shape[0]

        # ensemble is of size (20000, 256, 256) complex 128
        
        # tr_V_Us = np.einsum("mij,ij->m", ensemble, target.conj(), optimize=True)
        # tr_Us = np.einsum("aij,bij->ab", ensemble.conj(), ensemble, optimize=True)

        tr_V_Us = np.zeros((M, ), dtype=np.complex128)
        tr_Us = np.zeros((M, M), dtype=np.complex128)

        for i, un in enumerate(ensemble):
            trace_dist = np.trace(un @ target.conj().T)
            tr_V_Us[i] = trace_dist

        for i, un in enumerate(ensemble):
            for j, un2 in enumerate(ensemble):
                trace_dist = np.trace(un.conj().T @ un2)
                tr_Us[i, j] = trace_dist

        # f is of size (20000,)
        # H is of size (20000, 20000) floats 64
        f = -2 * np.real(tr_V_Us)
        H = 2 * np.real(tr_Us)

        # Make pos definite
        evs = np.linalg.eigvals(H)
        isposdef = np.all(evs > 0)
        trials = 0
        while not isposdef and trials < 20:
            H += 1e-10 * np.eye(M)
            print(f"Perturbing a little to make pos def try #: {trials}")
            evs = np.linalg.eigvals(H)
            isposdef = np.all(evs > 0)
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
        # s = -1 * np.linalg.inv(R) @ f
        probabilities = solve_qp(H, f, A=Aeq, b=beq, lb=lbound, ub=ubound, 
                                 solver='clarabel')   
        max = np.max(probabilities)
        min = np.min(probabilities)
        std = np.std(probabilities)
        print (f"Max prob: {max}, Min prob: {min}, Std prob: {std}", flush=True)
        return probabilities

    async def run(
            self, 
            circuit : Circuit, 
            data: PassData
    ) -> None:

        print("Running Generate Probability Pass", flush=True)
        checkpoint_dir = data["checkpoint_dir"]
        if self.run_on_ensemble_0:
            final_ens_file = f"{checkpoint_dir}/ensemble_0_.qasms"
            final_ens_jiggle_file = f"{checkpoint_dir}/ensemble_0_jiggles_.npy"
            final_ens_cache_file = f"{checkpoint_dir}/ensemble_cache_0.pkl"
        else:
            final_ens_file = f"{checkpoint_dir}/ensemble_final.qasms"
            final_ens_jiggle_file = f"{checkpoint_dir}/ensemble_final_jiggle.npy"
            final_ens_cache_file = f"{checkpoint_dir}/ensemble_cache_final.pkl"

        probs_file = f"{checkpoint_dir}/ensemble_final_probs.npy"


        if os.path.exists(probs_file):
            print("Already calculated probabilities, skipping", flush=True)
            return

        target = data.target
        if "final_unitaries" in data:
            print("Already Generated Unitaries")
            ensemble = data["final_unitaries"]
        else:
            try:
                circ_params = load_jiggled_ensemble(final_ens_file, 
                                                    final_ens_jiggle_file,
                                                    final_ens_cache_file)
            except:
                print("Corrupted ensemble files, skipping", checkpoint_dir, flush=True)
                return
            ensemble = await get_runtime().map(create_jiggled_unitaries, circ_params, 
                                               target=target, add_cost=False)
            ensemble = np.concatenate(ensemble, axis=0)

        if len(ensemble) > NUM_CIRCS_PER_PROB:
            rand_un_inds = np.random.choice(ensemble.shape[0], 
                                            size=NUM_CIRCS_PER_PROB, 
                                            replace=False)
            # Save random indices
            rand_inds_file = f"{checkpoint_dir}/ensemble_final_rand_inds.npy"
            np.save(rand_inds_file, rand_un_inds)
            ensemble = ensemble[rand_un_inds]

        print("Running Probaility on ensemble of size: ", ensemble.shape[0], flush=True)
            
        all_probs = GenerateProbabilityPass.calculate_probs(ensemble, 
                                                            target=target)

        if "checkpoint_dir" in data:
            np.save(probs_file, all_probs)
        return

