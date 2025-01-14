"""This module implements the InstantiateCount pass"""
from __future__ import annotations

from typing import Any

from bqskit.ir import Circuit
from bqskit.qis import UnitaryMatrix
from bqskit.ir.gates import CNOTGate
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
        success_threshold: float,
        size: int,
        min_chi_1: float = 0.3,
        min_chi_2: float = 0.5
    ) -> None:
        """
        Construct a Instantiate Count pass and then 

        """
        self.success_threshold = success_threshold
        self.size = size
        self.min_chi_1 = min_chi_1
        self.min_chi_2 = min_chi_2
        self.target = None
        return
    
    # async def calculate_chi_1_chi_2(self, ensemble: np.ndarray):
    #     num_reps = 50
    #     chi_1 = 0
    #     chi_2 = 0
    #     mean = np.mean(ensemble, axis=0)
    #     mean_epsi = 0

    #     for un in ensemble:
    #         diff = un - mean
    #         mean_epsi += np.abs(np.sum(np.einsum("ij,ij->", diff.conj(), diff)))
        
    #     mean_epsi /= len(ensemble)


    #     for _ in range(num_reps):
    #         if self.size > len(ensemble):
    #             print(f"How tf is this possible {len(ensemble)}")
    #             # print(ensemble[0])
    #             size = len(ensemble)
    #         else:
    #             size = self.size
    #         sub_ensemble_inds = np.random.choice(len(ensemble), size, replace=False)
    #         sub_ensemble = ensemble[sub_ensemble_inds]
    #         c_1, c_2 = get_chi_1_chi_2(sub_ensemble, mean=mean, mean_epsi=mean_epsi)
    #         chi_1 += c_1
    #         chi_2 += c_2

    #     return (chi_1 / num_reps, chi_2 / num_reps)
    
    def calculate_bias(self, ensemble: list[UnitaryMatrix], target: UnitaryMatrix):
        mean_un = np.mean(ensemble, axis=0)
        return normalized_frob_cost(mean_un, target)
    
    def calculate_e1(self, ensemble: list[UnitaryMatrix], target: UnitaryMatrix):
        e1s = [normalized_frob_cost(un, target) for un in ensemble]
        return np.mean(e1s)

    @staticmethod
    async def calculate_probs(ensemble: np.ndarray, target: np.ndarray):
        M = len(ensemble)

        tr_V_Us = np.zeros(M, dtype=np.complex128)
        tr_Us = np.zeros((M, M), dtype=np.complex128)

        print(ensemble.shape)

        for jj in range(M):
            tr_V_Us[jj] = np.trace(target.conj().T @ ensemble[jj])
            for kk in range(jj, M):
                a = np.trace(ensemble[jj].conj().T @ ensemble[kk])
                tr_Us[jj, kk] = a
                tr_Us[kk, jj] = a

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
        probs_file = f"{checkpoint_dir}/probs.data"

        if os.path.exists(probs_file):
            print("Already Generated Probs", flush=True)
            data["final_ensemble_probs"] = pickle.load(open(probs_file, "rb"))
            return
        
        best_ensemble: list[Circuit] = data["final_ensemble"]
        best_ensemble_unitaries: list[UnitaryMatrix] = np.array([circ.get_unitary() for circ in best_ensemble])

        if len(best_ensemble) < 5:
            data["final_ensemble_probs"] = [1 / len(best_ensemble) for _ in best_ensemble]
        else:
            # Now calculate the probability for this ensemble
            data["final_ensemble_probs"] = await GenerateProbabilityPass.calculate_probs(best_ensemble_unitaries, data.target)

        print("Calculated Probabilities", flush=True)

        if "checkpoint_dir" in data:
            pickle.dump(data["final_ensemble_probs"], open(probs_file, "wb"))
        return

