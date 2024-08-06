"""This module implements the InstantiateCount pass"""
from __future__ import annotations

import logging
from typing import Any

from bqskit.ir import Gate, Circuit
from bqskit.compiler.basepass import BasePass
from bqskit.runtime import get_runtime
from bqskit.compiler.passdata import PassData
import numpy as np
import matplotlib.pyplot as plt
from util import get_chi_1_chi_2
from qpsolvers import solve_ls

class GenerateProbabilityPass(BasePass):
    def __init__(
        self,
        size: int,
        min_chi_1: float = 0.3,
        min_chi_2: float = 0.5
    ) -> None:
        """
        Construct a Instantiate Count pass and then 

        """
        self.size = size
        self.min_chi_1 = min_chi_1
        self.min_chi_2 = min_chi_2
        return
    
    async def calculate_chi_1_chi_2(self, ensemble: np.ndarray):
        num_reps = 50
        chi_1 = 0
        chi_2 = 0
        mean = np.mean(ensemble, axis=0)
        mean_epsi = 0

        for un in ensemble:
            diff = un - mean
            mean_epsi += np.abs(np.sum(np.einsum("ij,ij->", diff.conj(), diff)))
        
        mean_epsi /= len(ensemble)


        for _ in range(num_reps):
            if self.size > len(ensemble):
                print(f"How tf is this possible {len(ensemble)}")
                # print(ensemble[0])
                size = len(ensemble)
            else:
                size = self.size
            sub_ensemble_inds = np.random.choice(len(ensemble), size, replace=False)
            sub_ensemble = ensemble[sub_ensemble_inds]
            c_1, c_2 = get_chi_1_chi_2(sub_ensemble, mean=mean, mean_epsi=mean_epsi)
            chi_1 += c_1
            chi_2 += c_2

        return (chi_1 / num_reps, chi_2 / num_reps)

    async def calculate_probs(self, ensemble: np.ndarray, target: np.ndarray):
        # For now return uniform
        # TODO: Implement Quadratic Program to 
        # return [1 / len(ensemble) for _ in ensemble]
        M = len(ensemble)

        tr_V_Us = np.zeros(M)
        tr_Us = np.zeros((M, M))

        print(ensemble.shape)

        for jj in range(M):
            tr_V_Us[jj] = np.trace(target.conj().T @ ensemble[jj])
            for kk in range(M):
                tr_Us[jj, kk] = np.trace(ensemble[jj].conj().T @ ensemble[kk])

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
        all_ensembles: list[list[Circuit]] = data["sub_select_ensemble"]
        all_ensemble_unitaries: list[np.ndarray] = [np.array([circ.get_unitary().numpy for circ in ensemble]) for ensemble in all_ensembles]

        data["ensemble_unitaries"] = all_ensemble_unitaries

        if len(all_ensembles) == 0:
            print("No ensembles to choose from")
            print(circuit)
            print(data.target)
            all_ensembles: list[list[Circuit]] = data["ensemble"]
            all_ensemble_unitaries: list[np.ndarray] = [np.array([circ.get_unitary().numpy for circ in ensemble]) for ensemble in all_ensembles]

        # For each ensemble, calculate chi_1 and chi_2
        chis = await get_runtime().map(self.calculate_chi_1_chi_2, all_ensemble_unitaries)

        ensemble_ind = 0
        best_chi = chis[0]

        # Select the best ensemble
        for i, chi in enumerate(chis):
            if chi > (self.min_chi_1, self.min_chi_2):
                best_chi = chi
                ensemble_ind = i
                break
            elif chi > best_chi:
                best_chi = chi
                ensemble_ind = i

        best_ensemble = all_ensembles[ensemble_ind]
        best_ensemble_unitaries = all_ensemble_unitaries[ensemble_ind]

        print("CHI_1", best_chi[0])
        print("CHI_2", best_chi[1])

        data["final_ensemble"] = best_ensemble

        # Now calculate the probability for this ensemble

        data["final_ensemble_probs"] = await self.calculate_probs(best_ensemble_unitaries, data.target)

        print("Calculated Probabilities")

        return





