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
# from .distance import frobenius_cost, normalized_frob_cost
# from qpsolvers import solve_qp
# import pickle
import os
import cvxpy as cp
import cvxopt

MAX_QP_CIRCS = 6000

class GenerateProbabilityPass(BasePass):
    
    def __init__(self, run_on_ensemble_0: bool = False, 
                 checkpoint_extra_str: str = "") -> None:
        self.run_on_ensemble_0 = run_on_ensemble_0
        self.checkpoint_extra_str = checkpoint_extra_str

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
            lp_prob.solve(solver=cp.CVXOPT) 

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
        else:
            ens_file = f"{checkpoint_dir}/ensemble_final.qasms"
            jiggle_file = f"{checkpoint_dir}/ensemble_final_jiggle.npy"
            cache_file = f"{checkpoint_dir}/ensemble_cache_final.pkl"
            # probs_file = f"{checkpoint_dir}/ensemble_final_probs.npy"

        final_probs_file = f"{checkpoint_dir}/ensemble_final_probs_{self.checkpoint_extra_str}.npy"

        if os.path.exists(final_probs_file):
            print("Already calculated probabilities, skipping", flush=True)
            return

        try:
            circ_params = load_jiggled_ensemble(ens_file, jiggle_file,
                                                cache_file, probs_file)
        except:
            print("Corrupted ensemble files, skipping", checkpoint_dir, flush=True)
            return

        target = data.target
        
        orig_circs = load_ensemble(ens_file)
        
        orig_uns = []
        all_caches = [c for _, _, _, c in circ_params]
        for i , c in enumerate(orig_circs):
            if all_caches[i] is not None:
                w_cache = get_runtime().get_cache()
                w_cache.clear()
                w_cache.update(all_caches[i])
            orig_uns.append(get_corrected_un(c.get_unitary(), target))

        ensemble = await get_runtime().map(create_jiggled_unitaries, circ_params, 
                                            target=target, add_cost=False)
        
        try:
            all_probs = np.load(probs_file)
        except:
            all_probs = None
            pass
        # To seed Frank-Wolf, we will first use Frank-Wolf on un-jiggled 
        # unitaries and then calculate the joint distribution of the full
        # ensemble

        orig_ensemble = np.array(orig_uns)
        orig_probs = GenerateProbabilityPass.calculate_probs(orig_ensemble,
                                                             target)
        
        # Calculate joint distribution for seeding FW
        if all_probs is not None:
            init_probs = [[p * qp_p for p in probs] for probs, qp_p in zip(all_probs, orig_probs)]
            init_probs = np.hstack(init_probs)
        else:
            init_probs = None

        full_ensemble = np.concatenate(ensemble, axis=0)

        print("Full ensemble shape: ", full_ensemble.shape, flush=True)

        if len(full_ensemble) > MAX_QP_CIRCS:
            print(f"Ensemble size {len(full_ensemble)} exceeds max {MAX_QP_CIRCS}, "
                  "shortening ensemble to reduce size", flush=True)
            # Pick random params per circ
            num_params_per_circ = ceil(MAX_QP_CIRCS / len(circ_params))

            # Pick inds with largest probabilities
            rand_inds = np.zeros((len(circ_params), num_params_per_circ), dtype=np.int64)
            if all_probs is not None:
                for i, p in enumerate(all_probs):
                    # Get n largest inds from p
                    sorted_inds = np.argsort(-p)
                    # Get the top num_params_per_circ indices
                    rand_inds[i, :] = sorted_inds[:num_params_per_circ]
            else:
                # If no probs, just pick random indices for each circ
                rand_inds = np.zeros((len(circ_params), num_params_per_circ), dtype=np.int64)
                for i, p in enumerate(all_probs):
                    rand_ind = np.random.choice(p.shape[0], size=num_params_per_circ, 
                                                replace=False)
                    rand_inds[i, :] = rand_ind

            # Pick the corresponding parameters from each item in circuits, params,
            # probs
            params = [p for _, p, _, _ in circ_params]
            # Choose only rand_un_inds params from each circ
            params = np.array([p[rand_inds[i], :] for i, p in enumerate(params)])
            # Choose rand_un_inds probs as well
            all_probs = np.array([p[rand_inds[i]] for i, p in enumerate(all_probs)])
            # Normalize the probabilities
            all_probs = all_probs / np.sum(all_probs, axis=1, keepdims=True)
            
            # Save the new ensemble
            store_params(params, jiggle_file)
            store_probs(all_probs, probs_file)

            # Recalculate the initial probabilities
            if init_probs is not None:
                init_probs = [[p * qp_p for p in probs] for probs, qp_p in zip(all_probs, orig_probs)]
                init_probs = np.hstack(init_probs)

            # Recalculate full_ensemble by choosing rand inds
            new_ensemble = []
            for i, circ_ensemble in enumerate(ensemble):
                new_circ_ensemble = [circ_ensemble[j] for j in rand_inds[i]]
                new_ensemble.append(new_circ_ensemble)
            full_ensemble = np.concatenate(new_ensemble, axis=0)

        print("Running Probability on ensemble of size: ", full_ensemble.shape[0], flush=True)
            
        all_probs = GenerateProbabilityPass.calculate_probs(full_ensemble, 
                                                            target=target,
                                                            initial_probs=init_probs)
        
        # Reshape all_probs to be of shape (num_circs, num_probs)
        num_circs = len(circ_params)
        all_probs = np.array(all_probs).reshape(num_circs, -1)

        if "checkpoint_dir" in data:
            np.save(final_probs_file, all_probs)
        return

