"""This module implements the InstantiateCount pass"""
from __future__ import annotations

from bqskit.ir import Circuit
from bqskit.runtime import get_runtime
from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
import numpy as np
from .common import load_jiggled_ensemble, create_jiggled_unitaries, load_ensemble
from .distance import get_corrected_un
# from .distance import frobenius_cost, normalized_frob_cost
# from qpsolvers import solve_qp
# import pickle
import os
import cvxpy as cp
import cvxopt

NUM_CIRCS_PER_PROB = 5000

class GenerateProbabilityPass(BasePass):
    
    def __init__(self, run_on_ensemble_0: bool = False) -> None:
        self.run_on_ensemble_0 = run_on_ensemble_0
        self.checkpoint_extra_str = ""



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

        final_probs_file = f"{checkpoint_dir}/ensemble_final_probs.npy"

        try:
            circ_params = load_jiggled_ensemble(ens_file, jiggle_file,
                                                cache_file, probs_file)
        except:
            print("Corrupted ensemble files, skipping", checkpoint_dir, flush=True)
            return

        # if os.path.exists(probs_file):
        #     print("Already calculated probabilities, skipping", flush=True)
        #     return

        target = data.target
        
        orig_circs = load_ensemble(ens_file)
        orig_uns = [get_corrected_un(c.get_unitary(), target) for c in orig_circs]
        ensemble = await get_runtime().map(create_jiggled_unitaries, circ_params, 
                                            target=target, add_cost=False)
        

        num_params_per_circ = circ_params[0][1].shape[1]
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

        ensemble = np.concatenate(ensemble, axis=0)

        # if len(ensemble) > NUM_CIRCS_PER_PROB:
        # rand_un_inds = np.random.choice(ensemble.shape[0], 
        #                                 size=NUM_CIRCS_PER_PROB, 
        #                                 replace=False)
        # Save random indices
        # rand_inds_file = f"{checkpoint_dir}/ensemble_final_rand_inds.npy"
        # np.save(rand_inds_file, rand_un_inds)
        # ensemble = ensemble[rand_un_inds]
        # init_probs = init_probs[rand_un_inds]
        # # Normalize init_probs
        # init_probs = init_probs / np.sum(init_probs)

        print("Running Probaility on ensemble of size: ", ensemble.shape[0], flush=True)
            
        all_probs = GenerateProbabilityPass.calculate_probs(ensemble, 
                                                            target=target,
                                                            initial_probs=init_probs)
        
        # Reshape all_probs to be of shape (num_circs, num_probs)
        num_circs = len(circ_params)
        all_probs = np.array(all_probs).reshape(num_circs, -1)

        if "checkpoint_dir" in data:
            np.save(final_probs_file, all_probs)
        return

