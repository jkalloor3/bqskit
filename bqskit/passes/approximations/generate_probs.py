"""This module implements the InstantiateCount pass"""
from __future__ import annotations

from bqskit.ir import Circuit
from bqskit.runtime import get_runtime
from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData

from bqskit.qis.unitary import UnitaryMatrix
import numpy as np
import os
import pickle
from math import ceil

import cvxpy as cp
import cvxopt

class GenerateProbabilitiesPass(BasePass):
    
    def __init__(self, 
                 max_qp_circs: int = 6000) -> None:
        """
        Generate the probability distribution by which to sample an ensemble.

        Args:
            max_qp_circs (int): The maximum number of circuits to run the 
            quadraticsolver on. The more circuits we consider, the slower the
            runtime.
        """
        self.max_qp_circs = max_qp_circs

    def calculate_probs(self, ensemble: np.ndarray, target: np.ndarray,
                        initial_probs: np.ndarray = None) -> np.ndarray:
        """Calculate the probabilities for the ensemble"""
        M = ensemble.shape[0]

        # ensemble is of size (20000, 256, 256) complex 128

        # Distance matrix to target (M, 1)
        tr_V_Us = np.einsum("mij,ij->m", ensemble, target.conj(), optimize=True)
        # Covariance matrix of ensemble (M, M)
        tr_Us = np.einsum("aij,bij->ab", ensemble.conj(), ensemble, 
                          optimize=True)

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
            # print('H not positive definite by a lot! Returning uniform dist')
            return [1 / len(ensemble) for _ in ensemble]
        
        # Constraints, probabilities should sum to 1 and be between 0 and 1
        Aeq = np.ones((1, M))
        beq = np.array([1])
        lbound = np.zeros(M)
        ubound = np.ones(M)
        
        ### FRANK WOLF SOLVER - Works better ... numerical stability? ###
        if initial_probs is None:
            probabilities = np.ones(M) / M
        else:
            probabilities = initial_probs

        nSteps = 20
        for _ in range(nSteps):
            DelJxk = H @ probabilities + f

            # Solve: min_y DelJxk.T @ y  s.t. Aeq @ y = beq, 
            # lbound <= y <= ubound
            y_var = cp.Variable(M)
            lp_obj = cp.Minimize(DelJxk @ y_var)
            lp_constraints = [Aeq @ y_var == beq, 
                              y_var >= lbound, 
                              y_var <= ubound]
            lp_prob = cp.Problem(lp_obj, lp_constraints)
            try:
                lp_prob.solve(solver=cp.CVXOPT)
            except:
                print("CVXOPT solver failed, returning probabilities early")
                return probabilities

            y = y_var.value
            step = y - probabilities

            # Optimal step size (gamma_star)
            numerator = -step @ DelJxk
            denominator = step @ H @ step

            if denominator > 1e-12:
                gamma_star = min(1.0, numerator / denominator)
            else:
                gamma_star = 1.0

            # Update
            probabilities = probabilities + gamma_star * step
        return probabilities

    
    def create_unitaries(
            self,
            ensemble: tuple[Circuit, np.ndarray],
            target: np.ndarray
    ) -> np.ndarray:
        circuit, params = ensemble
        # Get the unitary matrices for the ensemble
        unitary_matrices = [circuit.get_unitary(p) for p in params]
        # Stack them into a single array
        return np.array(unitary_matrices)

    async def run(
            self, 
            circuit : Circuit, 
            data: PassData
    ) -> None:
        # This pass should only be called if we are in ensemble mode
        assert "run_ensemble" in data and data["run_ensemble"] == True
        leap_file = str(data["ensemble_name"]) + "_leap_" + str(data["index"]) + ".pkl"
        circuits = pickle.load(open(leap_file, "rb"))
        probs_file = str(data["ensemble_name"]) + "_probs_" + str(data["index"]) + ".pkl"
        param_file = str(data["ensemble_name"]) + "_params_" + str(data["index"]) + ".pkl"



        if os.path.exists(probs_file):
            probs = pickle.load(open(probs_file, "rb"))
            # print(len(probs))
            # print(probs[0].shape, flush=True)
            data["ensemble_probabilities"] = probs
            return

        # Get ensemble circuits, params, and probabilities
        circuits: list[Circuit] = pickle.load(open(leap_file, "rb"))
        params: np.ndarray[float] = pickle.load(open(param_file, "rb"))
        init_probs: np.ndarray[float] = [np.ones((p.shape[0], )) / p.shape[0] for p in params]

        final_probs_3 = init_probs.copy()
        final_probs_3 /= np.sum(final_probs_3)

        # We have a list of M circuits, each with N unique parameters.
        # Our final probability distribution will be over all M*N circuits.
        M = len(circuits)
        N = params[0].shape[0]

        # We are using an iterative quadratic solver (Frank-Wolf) which takes in
        # an initial probability. For each set of N parameters, we have an 
        # initial probability distribution (init_probs). We need to 
        # create our best guess for the joint distribution over all M*N circuits

        # To do this, we are going to solve a smaller problem over M unitaries
        # which we can use to create the joint distribution. These M unitaries
        # will be the original M circuits with their corresponding parameters.
        orig_unitaries = np.array([c.get_unitary() for c in circuits])
        # Use QP to calculate outer probabilities
        outer_probs = self.calculate_probs(orig_unitaries, target=data.target)

        # Calculate joint distribution for seeding FW
        init_probs = np.array([[p * qp_p for p in probs] 
                              for probs, qp_p in zip(init_probs, 
                                                     outer_probs)])
        
        final_probs_2 = init_probs.copy()
        
        # print(init_probs.shape, flush=True)

        # If we have too many circuits,
        # we must clip the smallest probabilities to run FW on the rest
        if M * N > self.max_qp_circs:
            # Calculate a new N
            new_N = ceil(self.max_qp_circs / M)

            # Now choose the N highest probabilities for each circuit
            # If the probs are uniform, then choose random indices
            new_inds = np.ones((M, new_N), dtype=int) * -1
            for i, probs in enumerate(init_probs):
                std = np.std(probs)
                # Check if distribution is approximately uniform
                uniform = std < 1e-6
                if uniform:
                    # Get random indices
                    new_inds[i] = np.random.choice(len(probs), new_N, 
                                                   replace=False)
                else:
                    # Choose the indices of the highest probabilities
                    sorted_indices = np.argsort(probs)[-new_N:]
                    new_inds[i] = sorted_indices

            # Now choose the new params and probabilities
            params = [params[i][new_inds[i]] for i in range(M)]
            new_probs = [init_probs[i][new_inds[i]] for i in range(M)]
            # Make sure to normalize the new_probs
            init_probs /= np.sum(new_probs)

        # Zip together each circuit with corresponding parameters
        ensemble = list(zip(circuits, params))
        ensemble: list[list[UnitaryMatrix]] = await get_runtime().map(
            self.create_unitaries,
            ensemble, 
            target=data.target
        )

        # Format into a single list
        init_probs = np.hstack(init_probs)
        full_ensemble = np.concatenate(ensemble, axis=0)

        # print(full_ensemble.shape, init_probs.shape, flush=True)

        assert len(full_ensemble) == len(init_probs)
        
        final_probs_1 = self.calculate_probs(full_ensemble, 
                                           target=data.target,
                                           initial_probs=init_probs)
        
        # Reshape probs to be of size (M, N)
        final_probs_1 = np.array(final_probs_1).reshape(M, -1)

        print(final_probs_1.shape, final_probs_2.shape, final_probs_3.shape, flush=True)
        print(np.sum(final_probs_1), np.sum(final_probs_2), np.sum(final_probs_3), flush=True)

        # Store final params and probs
        data["ensemble_params"] = params
        data["ensemble_probabilities"] = final_probs_1
        # print(np.sum(final_probs_1), np.sum(final_probs_2), np.sum(final_probs_3))
        data["ensemble_probabilities_2"] = final_probs_2
        data["ensemble_probabilities_3"] = final_probs_3

        pickle.dump(data["ensemble_params"], open(param_file, "wb"))
        pickle.dump([final_probs_1, final_probs_2, final_probs_3], open(probs_file, "wb"))