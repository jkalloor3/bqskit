"""This module implements the InstantiateCount pass"""
from __future__ import annotations

from typing import Any

from bqskit.ir import Circuit
from bqskit.ir.gates import CNOTGate
from bqskit.compiler.basepass import BasePass
from bqskit.runtime import get_runtime
from bqskit.compiler.passdata import PassData
import numpy as np
from util import get_chi_1_chi_2
from qpsolvers import solve_ls

def cost(utry1: np.ndarray, utry2: np.ndarray, N: int) -> float:
    '''
    Calculates the normalized Frobenius distance between two unitaries
    '''
    diff = utry1- utry2
    # This is Frob(u - v)
    cost = np.real(np.trace(diff @ diff.conj().T))

    cost = cost / (N * N)

    # This quantity should be less than HS distance as defined by 
    # Quest Paper 
    return np.sqrt(cost)
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
    
    async def calculate_bias(self, ensemble: np.ndarray):
        mean_un = np.mean(ensemble, axis=0)
        return cost(mean_un, self.target, self.target.num_qudits)


    async def calculate_probs(self, ensemble: np.ndarray, target: np.ndarray):
        M = len(ensemble)

        tr_V_Us = np.zeros(M, dtype=np.complex128)
        tr_Us = np.zeros((M, M), dtype=np.complex128)

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

        # if "finished_probs_generation" in data:
        #     print("Already Generated Probs", flush=True)
        #     final_ensemble = data["final_ensemble"]
        #     data["final_ensemble_probs"] = [1 / len(final_ensemble) for _ in final_ensemble]
        #     return

        all_ensembles: list[list[Circuit]] = data["sub_select_ensemble"]
        all_ensemble_unitaries: list[np.ndarray] = [np.array([circ.get_unitary().numpy for circ in ensemble]) for ensemble in all_ensembles]

        data["ensemble_unitaries"] = all_ensemble_unitaries


        if len(all_ensembles) == 0:
            print("No ensembles to choose from")
            print(circuit)
            print(data.target)
            all_ensembles: list[list[Circuit]] = data["ensemble"]
            all_ensemble_unitaries: list[np.ndarray] = [np.array([circ.get_unitary().numpy for circ in ensemble]) for ensemble in all_ensembles]


        success_threshold = self.success_threshold * data["error_percentage_allocated"]
        self.target = data.target
        # For each ensemble, calculate the bias term
        biases = await get_runtime().map(self.calculate_bias, all_ensemble_unitaries)

        print("BIASES, ", biases)

        ensemble_ind = 0
        best_bias = biases[0]

        if best_bias > success_threshold ** 2:
            # Select the best ensemble
            for i, bias in enumerate(biases):
                if bias < best_bias:
                    best_bias = bias
                    ensemble_ind = i
                
                if bias < success_threshold ** 2:
                    break

        best_ensemble = all_ensembles[ensemble_ind]
        best_ensemble_unitaries = all_ensemble_unitaries[ensemble_ind]

        avg_cnots = np.mean([circ.count(CNOTGate()) for circ in best_ensemble])

        print("Orig CNOTS", circuit.count(CNOTGate()))
        print("Average CNOTS", avg_cnots)
        print("Bias", best_bias, "Threshold", success_threshold)

        data["final_ensemble"] = best_ensemble

        # Now calculate the probability for this ensemble

        data["final_ensemble_probs"] = await self.calculate_probs(best_ensemble_unitaries, data.target)

        print("Calculated Probabilities")

        # if "checkpoint_dir" in data:
        #     data["finished_probs_generation"] = True
        #     data.pop("sub_select_ensemble")
        #     checkpoint_data_file = data["checkpoint_data_file"]
        #     pickle.dump(data, open(checkpoint_data_file, "wb"))
        return





