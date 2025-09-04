"""This module implements the InstantiateCount pass"""
from __future__ import annotations

from bqskit.ir import Circuit
from bqskit.runtime import get_runtime
from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
import numpy as np
from math import ceil
from .common import (load_jiggled_ensemble, create_jiggled_unitaries, 
                     load_ensemble, store_params, store_probs)
from .distance import get_corrected_un
import os
import csv
import cvxpy as cp
import cvxopt

MAX_QP_CIRCS = 6000

class GenerateProbabilityPass(BasePass):
    
    def __init__(self, eps: float,
                  run_on_ensemble_0: bool = False, 
                 checkpoint_extra_str: str = "",) -> None:
        self.run_on_ensemble_0 = run_on_ensemble_0
        self.checkpoint_extra_str = checkpoint_extra_str

        factor = 4
        if eps < 10e-2:
            factor = 10
        else:
            factor = 50

        self.max_eps = eps * factor
        

    @staticmethod
    def calculate_probs(ensemble: np.ndarray, target: np.ndarray,
                        initial_probs: np.ndarray = None) -> np.ndarray:
        """Calculate the probabilities for the ensemble"""
        M = ensemble.shape[0]

        # ensemble is of size (20000, 256, 256) complex 128
        
        tr_V_Us = np.einsum("mij,ij->m", ensemble, target.conj(), optimize=True)
        tr_Us = np.einsum("aij,bij->ab", ensemble.conj(), ensemble, optimize=True)

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
        # probabilities = solve_qp(H, f, A=Aeq, b=beq, lb=lbound, ub=ubound, 
        #                          solver='clarabel')   
        
        ### FRANK WOLF SOLVER - Works better ... numerical stability? ###
        if initial_probs is None:
            probabilities = np.ones(M) / M
        else:
            probabilities = initial_probs

        nSteps = 20
        for _ in range(nSteps):
            DelJxk = H @ probabilities + f

            # Solve: min_y DelJxk.T @ y  s.t. Aeq @ y = beq, lbound <= y <= ubound
            # Using scipy.linprog for efficiency
            y_var = cp.Variable(M)
            lp_obj = cp.Minimize(DelJxk @ y_var)
            lp_constraints = [Aeq @ y_var == beq, y_var >= lbound, y_var <= ubound]
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
            gamma_star = min(1.0, numerator / denominator) if denominator > 1e-12 else 1.0

            # Update
            probabilities = probabilities + gamma_star * step

        p_max = np.max(probabilities)
        p_min = np.min(probabilities)
        p_std = np.std(probabilities)
        print (f"Max prob: {p_max}, Min prob: {p_min}, Std prob: {p_std}", flush=True)
        return probabilities

    async def run(
            self, 
            circuit : Circuit, 
            data: PassData
    ) -> None:

        print("Running Generate Probability Pass", flush=True)
        checkpoint_dir = data["checkpoint_dir"]
        if self.run_on_ensemble_0:
            ensemble_file_name = os.path.join(checkpoint_dir, "ensemble_{ind}_{extra}.qasms")
            jiggle_file_name = os.path.join(checkpoint_dir, "ensemble_{ind}_jiggles_{extra}.npy")
            probs_file_name = os.path.join(checkpoint_dir, "ensemble_{ind}_probs_{extra}.npy")
            cache_file_name = os.path.join(checkpoint_dir, "ensemble_{ind}_cache_{extra}.pkl")

            ens_file = ensemble_file_name.format(ind=0, extra=self.checkpoint_extra_str)
            jiggle_file = jiggle_file_name.format(ind=0, extra=self.checkpoint_extra_str)
            probs_file = probs_file_name.format(ind=0, extra=self.checkpoint_extra_str)
            cache_file = cache_file_name.format(ind=0, extra=self.checkpoint_extra_str)

        # We are given an ensemble of circuits, which correspond to M 
        # unique circuit structures, each with N jiggles for a total of MN circuits.

        # We are given an initial probability vector for each circuit structure
        # which is of size N

        # We have to generate a final probability for the entire MN ensemble. 
        # In order to do this, we will need an outer probability vector of size M
        # which is tensored with the inner probability vector of size N to get 
        # a probability vector of size MN.

        # This vector can then be passed to a Quadratic Program (QP) to optimize the probabilities.


        # Outer Probability Vector is Uniform - no QP at all
        # final_probs_file_1 = f"{checkpoint_dir}/ensemble_all_probs_1_{self.checkpoint_extra_str}.npy"
        # # Outer Probability Vector is according to QP on M unique circuits
        # final_probs_file_2 = f"{checkpoint_dir}/ensemble_all_probs_2_{self.checkpoint_extra_str}.npy"
        # # Outer Probability Vector is according to QP and then QP is run again on full ensemble
        # # In this case, the full ensemble may be shortened so that the QP can run
        # final_probs_file_3 = f"{checkpoint_dir}/ensemble_all_probs_3_{self.checkpoint_extra_str}.npy"

        # Check if CSV file exists and has bad data in it
        # checkpoint_data_file: str = data["checkpoint_data_file"]
        # final_csv_file = checkpoint_data_file.replace(".data", f"{self.checkpoint_extra_str}.csv")
        # rerun = False
        # if os.path.exists(final_csv_file):
        #     with open(final_csv_file, 'r') as file:
        #         reader = csv.DictReader(file)
        #         for row in reader:
        #             if "Epsilon" in row:  # Check if the column value is not empty
        #                 dist = float(row["Epsilon"])
        #                 if dist > self.max_eps:
        #                     # Bad ensemble, just rerun
        #                     rerun = True
        #                     break

        # if os.path.exists(final_probs_file_3) and not rerun:
        final_probs_no_qp_file = f"{checkpoint_dir}/ensemble_all_probs_no_qp_{self.checkpoint_extra_str}.npy"

        if os.path.exists(final_probs_no_qp_file):
            return

        try:
            circ_params = load_jiggled_ensemble(ens_file, jiggle_file,
                                                cache_file, probs_file)
            M = len(circ_params)
            N = circ_params[0][1].shape[0] # Params is of shape (N, num_params)

            print("Loaded ensemble with M = ", M, " and N = ", N, flush=True)
        except:
            return

        # target = data.target
        
        # orig_circs = load_ensemble(ens_file)
        
        # orig_uns = []
        # all_caches = [c for _, _, _, c in circ_params]
        # for i , c in enumerate(orig_circs):
        #     if all_caches[i] is not None:
        #         w_cache = get_runtime().get_cache()
        #         w_cache.clear()
        #         w_cache.update(all_caches[i])
        #     orig_uns.append(get_corrected_un(c.get_unitary(), target))

        # ensemble_dists: list[list[tuple[np.ndarray, float]]] = await get_runtime().map(create_jiggled_unitaries, circ_params, 
        #                                     target=target, add_cost=True,
        #                                     drop_zeros=False)

        # ensemble = [[x[0] for x in sub_ens_dist] for sub_ens_dist in ensemble_dists]
        # dists = np.array([[x[1] for x in sub_ens_dist] for sub_ens_dist in ensemble_dists])

        try:
            all_probs = np.load(probs_file)
            uniform = False
        except:
            # Then assume initial probabilities are uniform for each set of N
            all_probs = [np.ones(N) / N for _ in range(M)]
            uniform = True
            pass

        # For any dist > 100 * eps, set the corresponding prob to 0 and renormalize
        # for i, probs in enumerate(all_probs):
        #     bad_dist_inds = np.where(dists[i] > self.max_eps)[0]
        #     if len(bad_dist_inds) > 0:
        #         print(f"Bad dist indices for circuit {i}: {bad_dist_inds}", flush=True)
        #         # Set the probabilities to 0 for these indices
        #         probs[bad_dist_inds] = 0.0
        #         # Renormalize the probabilities
        #         if np.sum(probs) == 0:
        #             print(f"All probabilities for circuit {i} are 0, setting to lowest distance - BAD ENSEMBLE", flush=True)
        #             min_ind = np.argmin(dists[i])
        #             probs[min_ind] = 1.0

        #         probs /= np.sum(probs)
        #     all_probs[i] = probs

        # To seed Frank-Wolf, we will first use Frank-Wolf on un-jiggled 
        # unitaries and then calculate the joint distribution of the full
        # ensemble

        # orig_ensemble = np.array(orig_uns)
        # # Uniform outer probabilities for original ensemble
        outer_probs_1 = np.ones(M) / M
        # # Use QP to calculate outer probabilities
        # outer_probs_2 = GenerateProbabilityPass.calculate_probs(orig_ensemble,
        #                                                      target)
        
        # Calculate joint distribution for seeding FW
        final_probs_no_qp = np.array([[p * qp_p for p in probs] for probs, qp_p in zip(all_probs, outer_probs_1)])
        store_probs(final_probs_no_qp, final_probs_no_qp_file)
        # final_probs_2 = [[p * qp_p for p in probs] for probs, qp_p in zip(all_probs, outer_probs_2)]
        # Calculate the initial probabilities for the last QP pass
        # init_probs = np.hstack(final_probs_2)
        # if M * N > MAX_QP_CIRCS:
        #     # Choose a new N such that M * N <= MAX_QP_CIRCS
        #     new_N = ceil(MAX_QP_CIRCS / M)
        #     print(f"Reducing ensemble size to {M} * {new_N} = {M * new_N} for QP", flush=True)

        #     # Now choose the N highest probabilities for each circuit
        #     # If the probs are uniform, then choose random indices
        #     new_inds = np.ones((M, new_N), dtype=int) * -1
        #     for i, probs in enumerate(all_probs):
        #         if uniform:
        #             # Get random indices
        #             new_inds[i] = np.random.choice(len(probs), new_N, 
        #                                            replace=False)
        #         else:
        #             # Choose the indices of the highest probabilities
        #             sorted_indices = np.argsort(probs)[-new_N:]
        #             new_inds[i] = sorted_indices

        #     # Now choose the new params and probabilities
        #     old_params = np.array([params for _, params, _, _ in circ_params])
        #     old_probs = np.array([probs for _, _, probs, _ in circ_params])
        #     new_params = np.array([old_params[i][new_inds[i]] for i in range(M)])
        #     new_probs = np.array([old_probs[i][new_inds[i]] for i in range(M)])
        #     # Make sure to normalize the new_probs for each row
        #     new_probs = new_probs / np.sum(new_probs, axis=1, keepdims=True)
        #     # Multiply by outer probability vector
        #     init_probs = [[p * qp_p for p in probs] for probs, qp_p in zip(new_probs, outer_probs_2)]
        #     init_probs = np.hstack(init_probs)

        #     # Calculate new circ_params
        #     new_circ_params = []
        #     for i in range(M):
        #         new_circ_params.append((orig_circs[i], new_params[i], 
        #                                 new_probs[i], all_caches[i]))
                
        #     print("Init Probabilities shape: ", init_probs.shape, flush=True)
        #     print("New Parameters shape: ", new_params.shape, flush=True)
                
        #     # Calculate the new ensemble
        #     ensemble = await get_runtime().map(create_jiggled_unitaries, 
        #                                     new_circ_params, target=target,
        #                                     drop_zeros=False, 
        #                                     add_cost=False)
            
        #     # Now save the new params file to use in next pass
        #     new_jiggle_file = os.path.join(checkpoint_dir, f"ensemble_0_jiggles_{self.checkpoint_extra_str}_sub.npy")
        #     np.save(new_jiggle_file, new_params)
            
        # full_ensemble = np.concatenate(ensemble, axis=0)
        # print("Full ensemble shape: ", full_ensemble.shape, flush=True)


        # print("Running Probability on ensemble of size: ", full_ensemble.shape[0], flush=True)
        
        # assert full_ensemble.shape[0] == len(init_probs), \
        #     f"Full ensemble size does not match initial probabilities size {checkpoint_dir}"
            
        # final_probs_3 = GenerateProbabilityPass.calculate_probs(full_ensemble, 
        #                                                     target=target,
        #                                                     initial_probs=init_probs)
    
        # print("Sum of all probs: ", np.sum(final_probs_1), np.sum(final_probs_2), np.sum(final_probs_3), flush=True)
        
        # # Reshape all_probs to be of shape (M, N)
        # final_probs_3 = np.array(final_probs_3).reshape(M, -1)

        # # Store all probabilities
        # store_probs(final_probs_1, final_probs_file_1)
        # store_probs(final_probs_2, final_probs_file_2)
        # store_probs(final_probs_3, final_probs_file_3)

