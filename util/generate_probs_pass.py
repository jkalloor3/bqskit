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
from .distance import frobenius_cost, normalized_frob_cost
from qpsolvers import solve_ls
import pickle
import os


class GenerateProbabilityPass(BasePass):
    
    def __init__(
        self,
    ) -> None:
        """
        Construct a Instantiate Count pass and then 

        """
        return
    
    @staticmethod
    def calculate_probs(ensemble: np.ndarray, target: np.ndarray) -> np.ndarray:
        M = len(ensemble)

        # tr_V_Us = np.zeros(M, dtype=np.complex128)
        # tr_Us = np.zeros((M, M), dtype=np.complex128)

        print(ensemble.shape, flush=True)

        tr_V_Us = np.einsum("mij,ij->m", ensemble, target.conj())
        tr_Us = np.einsum("aij,bij->ab", ensemble.conj(), ensemble)

        # for jj in range(M):
        #     utry = ensemble[jj]
        #     for kk in range(jj, M):
        #         a = np.einsum("ij,ij->", ensemble[kk].conj(), utry)
        #         tr_Us[jj, kk] = a
        #         tr_Us[kk, jj] = a

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

        return probabilities

    async def run(
            self, 
            circuit : Circuit, 
            data: PassData
    ) -> None:

        print("Running Generate Probability Pass", flush=True)
        checkpoint_dir = data["checkpoint_dir"]
        probs_file = f"{checkpoint_dir}/ensemble_final_probs.npy"

        # if os.path.exists(probs_file):
        #     print("Already Generated Probs", flush=True)
        #     data["final_ensemble_probs"] = np.load(probs_file)
        #     return
        
        best_ensemble_unitaries: list[UnitaryMatrix] = data["final_ensemble_unitaries"]
        best_ensemble_unitaries = np.stack([x.numpy for x in best_ensemble_unitaries])
        # best_ensemble_unitaries: list[UnitaryMatrix] = np.array([circ.get_unitary() for circ in best_ensemble])

        print(f"Calculating Probs on {len(best_ensemble_unitaries)} unitaries", flush=True)

        if len(best_ensemble_unitaries) < 5:
            data["final_ensemble_probs"] = [1 / len(best_ensemble_unitaries) for _ in best_ensemble_unitaries]
        else:
            # Now calculate the probability for this ensemble
            data["final_ensemble_probs"] = GenerateProbabilityPass.calculate_probs(best_ensemble_unitaries, data.target)

        print("Calculated Probabilities", flush=True)

        if "checkpoint_dir" in data:
            np.save(probs_file, data["final_ensemble_probs"])
        return

