"""This module implements the ToU3Pass."""
from __future__ import annotations

import logging

from pathlib import Path

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.passes import ToU3Pass, ForEachBlockPass
from bqskit.ir.circuit import Circuit, CircuitPoint
from bqskit.runtime import get_runtime
from typing import Any
from itertools import chain, product
from bqskit.ir.lang.qasm2 import OPENQASM2Language
from bqskit.ir.opt.cost.functions import GPNormalizedFrobeniusCostGenerator, GPNormalizedFrobeniusCostGenerator
from bqskit.ir.opt.minimizers.lbfgs import LBFGSMinimizer
from bqskit.ir.opt.cost.generator import CostFunctionGenerator
from bqskit.ir.gates import *
from bqskit.qis import UnitaryMatrix
import numpy as np
from math import ceil, floor
import pickle
import time

from bqskit.qis.pauli import PauliMatrices
from bqskit.ir import Circuit
from bqskit.qis import UnitaryMatrix
import scipy as sp
from bqskit.utils.math import dot_product
from bqskit.runtime import get_runtime

import os

from util.distance import get_corrected_un, normalized_gp_frob_cost
from .common import (store_params, load_ensemble_strs, store_ensemble_strs, store_probs)
from .counter import count_params_str
from .gg import GridSynthGate, gg_gate_def, MIN_EPSILON, get_rz_perturbations, get_approx_t_str

_logger = logging.getLogger(__name__)

frob_cost = GPNormalizedFrobeniusCostGenerator()
lang = OPENQASM2Language(gate_defs=[("gg", gg_gate_def)])

MAX_GGS_TO_JIGGLE = 5
MIN_PROBABILITY = 1e-8


def split_float_to_int(value: float, num: int) -> list[int]:
    """Split a float into a list of integers that average to the float."""

    initial_arr = np.ones(num) * floor(value)
    ind = num - 1
    while np.mean(initial_arr) < value:
        initial_arr[ind] += 1
        ind -= 1
        ind = ind % num
        
    return initial_arr.astype(int).tolist()


class  JiggleEnsemblePass(BasePass):
    """Converts single-qubit general unitary gates to U3 Gates."""
    num_jiggles = 0

    finished_pass_str = "finished_jiggle"

    def __init__(self, success_threshold = 1e-4, 
                 num_circs = 100,
                 cost: CostFunctionGenerator = GPNormalizedFrobeniusCostGenerator(),
                 use_ensemble: bool = True,
                 use_scan_sols: bool = False,
                 use_calculated_error: bool = True,
                 count_t: bool = False,
                 checkpoint_extra_str: str = "",
                 jiggle_skew: int = 0,
                 do_u3_perturbation: bool = True,
                 pass_ensemble: bool = False,
                 flood_circ: bool = False) -> None:
        """
        Construct a ToU3Pass.

        Args:
            convert_all_single_qubit_gates (bool): Indicates wheter to convert
            only the general gates, or every single qubit gate.
        """

        self.success_threshold = success_threshold
        self.num_circs = num_circs
        self.cost = cost
        self.use_ensemble = use_ensemble
        self.use_scan_sols = use_scan_sols
        self.pass_ensemble = pass_ensemble
        self.instantiate_options: dict[str, Any] = {
            'dist_tol': self.success_threshold,
            'min_iters': 100,
            'cost_fn_gen': self.cost,
            'method': 'minimization',
            'minimizer': LBFGSMinimizer(),
        }
        self.use_calculated_error = use_calculated_error
        self.count_t = count_t
        self.checkpoint_extra_str = checkpoint_extra_str
        self.jiggle_skew = jiggle_skew
        self.do_u3_perturbation = do_u3_perturbation
        self.do_flood_circ = flood_circ
        assert not (self.use_ensemble and self.use_scan_sols), \
            "Cannot use both ensemble and scan solutions at the same time."

    @staticmethod
    def get_perturbations(epsilon: float, ens_size: int) -> list[UnitaryMatrix]:
        perturbations = []
        pauli_strings = ["X", "Y", "Z"]
        paulis = [PauliMatrices.from_string(pauli) for 
                  pauli in pauli_strings]
        
        all_coeffs = []
        for _ in range(ens_size // 2):
            all_coeffs.append(np.random.rand(len(paulis)))
            
        for coeff in all_coeffs:
            coeff /= np.linalg.norm(coeff)
            coeff *= epsilon
            H_1 = dot_product(coeff, paulis)
            H_2 = dot_product(-1 * coeff, paulis)
            assert np.allclose(H_1, H_1.conj().T)
            assert np.allclose(H_2, H_2.conj().T)
            eiH_1 = sp.linalg.expm(1j * H_1)
            eiH_2 = sp.linalg.expm(1j * H_2)
            perturbations.append(UnitaryMatrix(eiH_1))
            perturbations.append(UnitaryMatrix(eiH_2))
        
        return perturbations

    @staticmethod
    def get_ham_perturbations(u3_utry: UnitaryMatrix, dist: float, num_options: int) -> list[list[float]]:
        perturbations = JiggleEnsemblePass.get_perturbations(dist, num_options)
        final_matrices = [u3_utry @ perturbation for perturbation in perturbations]
        final_params = [U3Gate().calc_params(mat) for mat in final_matrices]
        # corrected_uns = [get_corrected_un(mat, u3_utry) for mat in final_matrices]
        # avg_un = np.mean(corrected_uns, axis=0)
        # bias_cost = normalized_gp_frob_cost(avg_un, u3_utry)
        # eps = np.mean([normalized_gp_frob_cost(mat, u3_utry) for mat in final_matrices])
        # print("Local Bias Cost: ", bias_cost, "Epsilon: ", eps, flush=True)
        avg_dist = np.mean([normalized_gp_frob_cost(final_mat, u3_utry) for final_mat in final_matrices])
        return final_params, avg_dist


    @staticmethod
    def get_all_gg_params(circs: list[str], target: UnitaryMatrix, 
                          success_threshold: float) -> list[tuple[list[
                              list[list[float]]], list[list[float]], dict]]:
        '''
        Given a set of circuits, get all the GG params, probabilities, 
        and local cache for Gridsynth Strings.

        Additionally, output the maximum number of circuits we use to size
        everything in the next pass
        '''
        all_data = []
        for circ_str in circs:
            circ = lang.decode(circ_str)
            dist = frob_cost.calc_cost(circ, target)
            # For each U3 gate, calculate do a Hamiltonian perturbation
            num_u3s = circ.count(U3Gate())
            assert num_u3s == 0
            num_ggs = circ.count(GridSynthGate())
            if num_ggs == 0:
                # Return a num x 1 array of zeros
                all_data.append((None, None, {}))
                continue

            if dist > success_threshold:
                # print("Dist greater than success threshold", flush=True)
                all_data.append((None, None, {}))
                continue

            # Map GG angles to potential GG params and probabilities
            # For each GG, keep track of list of possible params [angle, eps, z]
            gg_param_options: list[list[list[float]]] = []
            # Additionally keep track of probabilities for each GG
            gg_probs: list[list[float]] = []

            # Limit number of GGs to jiggle
            if num_ggs > MAX_GGS_TO_JIGGLE:
                num_ggs = MAX_GGS_TO_JIGGLE

            orig_perturb_dist = success_threshold - dist
            perturb_dist = orig_perturb_dist / (num_ggs + 1)
            # Round log of dist to nearest int
            perturb_dist = -1 * np.log10(perturb_dist)

            int_perturb_dists = split_float_to_int(perturb_dist, num_ggs) 

            # Only need to save the GG strings
            cache_to_save = {}
            # Reuse params and probabilities if possible
            total_cache = {}
            for op in circ.operations():
                if isinstance(op.gate, GridSynthGate):
                    # Fill up cache with orig params
                    ind_orig = (op.params[0], int(op.params[1]))
                    if ind_orig not in cache_to_save:
                        cache_to_save[ind_orig] = get_approx_t_str(op.params[0], int(op.params[1]))

                    if len(gg_probs) < num_ggs:
                        angle = op.params[0]
                        gg_params, gg_strs, probs = total_cache.get(angle, get_rz_perturbations(angle, 
                                                                        int_perturb_dists[len(gg_param_options)]))
                        
                        if gg_params is None: # Bad Scaling Factor, don't use
                            gg_param_options.append([None])
                            gg_probs.append([1.0])
                            continue
                        
                        total_cache[angle] = (gg_params, gg_strs, probs)
                            
                        # Fill up cache to save with gg_strs
                        for i, gg_str in enumerate(gg_strs):
                            ind = (gg_params[i][0], gg_params[i][1])
                            cache_to_save[ind] = gg_str

                        gg_param_options.append(gg_params)
                        gg_probs.append(probs)

            all_data.append((gg_param_options, gg_probs, cache_to_save))

        return all_data

    @staticmethod
    def single_jiggle_ham_clifft(circ_data: tuple[str, list, list], 
                                 num: int) -> tuple[
                                     np.ndarray[float], 
                                     np.ndarray[float]]:
        '''
        Apply Hamiltonian perturbations to the circuit.

        Input:
        circ_data: circ_str, gg_param_options, gg_probs

        num: Size we need to create for final params and probabilities
        
        Returns:
        A numpy array of size (num, num_params) where num_params is the number 
        of parameters in the circuit

        Also returns a probabilitiy vector of size (num_params, ) corresponding
        to the probability of using that circuit and parameters. We clip the 
        probability at `MIN_PROBABILITY`.
        
        '''

        circ_str, gg_param_options, gg_probs = circ_data
        circ = lang.decode(circ_str)

        if gg_param_options is None:
            # return array of (circ_params) * num
            empty_params = np.vstack([circ.params] * num)
            empty_probs = np.ones((num, )) / num
            return empty_params, empty_probs

        circ = lang.decode(circ_str)
        # This selects which gg str we should use
        all_gg_inds = list(product(*[range(len(probs)) for probs in gg_probs]))

        final_params = np.zeros((num, circ.num_params))
        final_probs = np.zeros((num, ))

        num_params = circ.num_params

        for i, selections in enumerate(all_gg_inds):
            new_circ = circ.copy()
            gg_ind = 0
            gg_prob = [gg_probs[j][selection] for j, selection in enumerate(selections)]
            gg_params= [gg_param_options[j][selection] for j, selection in enumerate(selections)]
            total_prob = np.prod(gg_prob)
            if total_prob < MIN_PROBABILITY:
                final_params[i, :] = new_circ.params
                final_probs[i] = 0
                continue

            for op in new_circ.operations():
                if gg_ind >= len(gg_prob):
                    continue
                if isinstance(op.gate, GridSynthGate):
                    new_gg_param = gg_params[gg_ind]
                    if new_gg_param is not None:
                        op.params = new_gg_param
                    # Otherwise, use original params
                    gg_ind += 1

            assert new_circ.num_params == num_params
            
            final_params[i, :] = new_circ.params
            final_probs[i] = total_prob

        return final_params, final_probs
    
    @staticmethod
    def single_jiggle_ham(circ_str: str, 
                          target: UnitaryMatrix, 
                          num: int, 
                          success_threshold: float) -> np.ndarray[float]:
        '''
        Apply Hamiltonian perturbations to the U3 gates in the circuit.

        Input:
        circ_str

        circ_str is the string representation of the circuit

        We require the circ_num to create a local cache for Gridsynth Strings

        target:
        target is the target unitary matrix

        num: number of circuits to generate

        success_threshold: success threshold for the perturbation
        

        Returns:
        A numpy array of size (num, num_params) where num_params is the number 
        of parameters in the circuit
        '''
        circ = lang.decode(circ_str)
        circ_id = hash(circ_str) % 100
        dist = frob_cost.calc_cost(circ, target)
        # For each U3 gate, calculate do a Hamiltonian perturbation
        num_u3s = circ.count(U3Gate())
        # print("Init Dist: ", dist, flush=True)
        if (num_u3s) == 0:
            print("No U3s", circ.gate_counts, flush=True)
            # Repeat circ params num*2 times
            empty_params = np.vstack([circ.params] * (num * 2))
            return empty_params
        
        if dist > 2 * success_threshold:
            return None

        # For each u3, come up with 4 perturbations
        orig_perturb_dist = success_threshold - dist
        perturb_dist = (orig_perturb_dist / (MAX_GGS_TO_JIGGLE))
        perturb_dist = JiggleEnsemblePass.calculate_perturb_dist(perturb_dist,
                                                                 circ_str,
                                                                 target,
                                                                 success_threshold)

        print(f"{circ_id}: Using Perturb Dist: ", perturb_dist, flush=True)
        num_options = 2

        # Generate param options
        u3_param_options: dict[tuple[float, float, float], 
                            list[list[float]]] = {}
        
        for cycle, op in circ.operations_with_cycles():
            if isinstance(op.gate, U3Gate):
                # Round params to int_perturb_dist
                key = (tuple(np.round(op.params)), cycle, op.location[0])
                options, avg_dist = JiggleEnsemblePass.get_ham_perturbations(op.get_unitary(), perturb_dist, num_options * 2)
                u3_param_options[key] = options

            if len(u3_param_options) >= MAX_GGS_TO_JIGGLE:
                break

        # Now sample over all possible combinations
        all_inds = list(product(*[range(len(options)) for options in u3_param_options.values()]))
        final_params = np.zeros((len(all_inds), circ.num_params))
        avg_dist = 0.0
        avg_un = np.zeros_like(target.numpy)
        for i, selections in enumerate(all_inds):
            new_circ = circ.copy()
            ind = 0
            for op in new_circ.operations():
                if ind >= len(selections):
                    continue
                if isinstance(op.gate, U3Gate):
                    key = (tuple(np.round(op.params)), cycle, op.location[0])
                    if key in u3_param_options:
                        new_u3_param = u3_param_options[key][selections[ind]]
                        op.params = new_u3_param
                        ind += 1
            
            final_params[i, :] = new_circ.params
            avg_dist += (frob_cost.calc_cost(new_circ, target) / len(all_inds))
            avg_un += (get_corrected_un(new_circ.get_unitary(), circ.get_unitary()) / len(all_inds))

        bias = normalized_gp_frob_cost(avg_un, target)
        print(f"{circ_id}: Avg Dist: ", avg_dist, "Bias: ", bias, "Init Dist: ", dist, flush=True)

        # avg_dist = total_dist / (num * 2)
        # avg_un /= (num * 2)
        # bias_cost = normalized_gp_frob_cost(avg_un, target)
        # print(f"{circ_id}: Bias Cost: ", bias_cost, "Eps: ", avg_dist, flush=True)
        # if avg_dist > 2 * success_threshold:   
        #     print(f"{circ_id}: Avg Dist too high: ", avg_dist, flush=True)
        #     return JiggleEnsemblePass.slow_jiggle_ham_outer(circ_str, 
        #                                               target, 
        #                                               num, 
        #                                               success_threshold,
        #                                               perturb_dist / 2)
        # elif avg_dist < 0.25 * success_threshold:
        #     print(f"{circ_id}: Avg Dist very low: ", avg_dist, flush=True)
        #     return JiggleEnsemblePass.slow_jiggle_ham_outer(circ_str,
        #                                                 target, 
        #                                                 num, 
        #                                                 success_threshold,
        #                                                 perturb_dist * 1.5)
        # else:
        #     print(f"{circ_id}: Final Avg Dist: ", avg_dist, flush=True)

        return final_params
    

    @staticmethod
    def calculate_perturb_dist(perturb_dist: float,
                               circ_str: str,
                                 target: UnitaryMatrix,
                                 success_threshold: float) -> float:
        ''' Calculate appropriate perturbation distance '''

        circ = lang.decode(circ_str)

        while True:
            out_distance = JiggleEnsemblePass.get_perturbation_distance(circ,
                                                                        target,
                                                                        perturb_dist)
            if out_distance > 2 * success_threshold:
                perturb_dist /= 2
            elif out_distance < success_threshold:
                perturb_dist *= 1.5
            else:
                return perturb_dist
            
        

    @staticmethod
    def get_perturbation_distance(circ: Circuit,
                                  target: UnitaryMatrix,
                                    perturb_dist: float) -> float:
        ''' Get single shot perturbation distance '''
        new_circ = circ.copy()
        num_jiggles_left = MAX_GGS_TO_JIGGLE
        for op in new_circ.operations():
            if isinstance(op.gate, U3Gate):
                p_param = JiggleEnsemblePass.get_ham_perturbations(op.get_unitary(),
                                                                   perturb_dist, 
                                                                   2, )[0][0]
                op.params = p_param
                num_jiggles_left -= 1
            if num_jiggles_left <= 0:
                break

        dist = frob_cost.calc_cost(new_circ, target)
        return dist
            
    '''
    # def slow_jiggle_ham_outer(circ_str: str,
    #                           target: str,
    #                           num: int,
    #                           success_threshold: float,
    #                           perturb_dist: float) -> np.ndarray[float]:
    #     
    #     Keep calling slow jiggle ham with increasing perturb dist until
    #     we can get a good set of jiggled circuits.
    #     
    #     max_tries = 20
    #     circ_id = hash(circ_str) % 100

    #     while max_tries > 0:
    #         final_params, avg_dist = JiggleEnsemblePass.slow_jiggle_ham(circ_str, 
    #                                                                     target, 
    #                                                                     num, 
    #                                                                     success_threshold, 
    #                                                                     perturb_dist)
    #         if avg_dist > 2 *success_threshold:
    #             perturb_dist /= 2
    #         elif avg_dist < 0.25 * success_threshold:
    #             perturb_dist *= 1.5
    #         else:
    #             print(f"{circ_id}: Final Avg Dist after slow jiggle: ", avg_dist, flush=True)
    #             return final_params
    #         max_tries -= 1
    #     return final_params

    # @staticmethod
    # def slow_jiggle_ham(circ_str: str, 
    #                     target: str, 
    #                     num: int, 
    #                     success_threshold: float,
    #                     perturb_dist: float) -> np.ndarray[float]:
    #     
    #     If the previous jiggle failed, then slowly add perturbations until
    #     we can get a good set of jiggled circuits.
    #     

    #     circ = lang.decode(circ_str)
    #     circ_id = hash(circ_str) % 100
    
    #     # Start with smaller this perturb dist
    #     # Generate param options
    #     u3_param_options: dict[tuple[float, float, float], np.ndarray[float]] = {}
    #     num_options = 4
    #     sample_circ = circ.copy()
    #     for cycle, op in circ.operations_with_cycles():
    #         pt = (cycle, op.location[0])
    #         if isinstance(op.gate, U3Gate):
    #             # Round params to int_perturb_dist
    #             key = (tuple(np.round(op.params)), pt)
    #             if key not in u3_param_options:
    #                 options, _ = JiggleEnsemblePass.get_ham_perturbations(op.get_unitary(), perturb_dist, num_options * 2)
    #                 u3_param_options[key] = options
    #             else:
    #                 options = u3_param_options[key]
    #             sample_un = ConstantUnitaryGate(U3Gate().get_unitary(options[0]))
    #             sample_circ.replace_gate(CircuitPoint(pt), sample_un, op.location)
                
    #             # Check if dist < threshold still
    #             new_dist = frob_cost.calc_cost(sample_circ, target)
    #             if new_dist > success_threshold:
    #                 # Break out of while loop if too large
    #                 print(f"{circ_id}: Final Dist, ending at: ", new_dist, len(u3_param_options), flush=True)
    #                 break
    #             # else:
    #             #     print("New Dist: ", new_dist, len(u3_param_options), flush=True)

    #     final_params = []
    #     new_circ = circ.copy()
    #     total_dist = 0.0
    #     num_u3s = circ.count(U3Gate())
    #     for _ in range(num):
    #         new_circ.set_params(circ.params)
    #         u3_ind = 0
    #         z_ind = 0
    #         rand_u3_inds = np.random.choice(num_options, num_u3s, replace=True)
    #         for cycle, op in new_circ.operations_with_cycles():
    #             pt = (cycle, op.location[0])
    #             if isinstance(op.gate, U3Gate):
    #                 key = (tuple(np.round(op.params)), pt)
    #                 if key in u3_param_options:
    #                     op.params = u3_param_options[key][rand_u3_inds[u3_ind] * 2]
    #                     u3_ind += 1


    #         full_params_1 = new_circ.params
    #         u3_ind = 0
    #         new_circ.set_params(circ.params)
    #         for cycle, op in new_circ.operations_with_cycles():
    #             pt = (cycle, op.location[0])
    #             if isinstance(op.gate, U3Gate):
    #                 key = (tuple(np.round(op.params)), pt)
    #                 if key in u3_param_options:
    #                     op.params = u3_param_options[key][rand_u3_inds[u3_ind] * 2 + 1]
    #                     u3_ind += 1

    #         full_params_2 = new_circ.params
    #         final_params.append(full_params_1)
    #         final_params.append(full_params_2)
    #         total_dist += frob_cost.calc_cost(new_circ, target)

    #     avg_dist = total_dist / (num)
    #     return np.vstack(final_params), avg_dist


    '''
    
    @staticmethod
    def flood_circ(circ_str: str) -> str:
        circ = lang.decode(circ_str)
        if circ.count(U3Gate()) > MAX_GGS_TO_JIGGLE:
            print("Already enough U3s, not flooding", flush=True)
            return circ_str
        num_u3s_needed = MAX_GGS_TO_JIGGLE - circ.count(U3Gate())
        flooded_cnot = Circuit(2)
        flooded_cnot.append_gate(CNOTGate(), (0, 1))
        flooded_cnot.append_gate(U3Gate(), (0,), [0, 0, 0])
        flooded_cnot.append_gate(U3Gate(), (1,), [0, 0, 0])
        init_u3_circ = Circuit(circ.num_qudits)
        for i in range(circ.num_qudits):
            init_u3_circ.append_gate(U3Gate(), (i,), [0, 0, 0])
        circ.insert_circuit(0, init_u3_circ, tuple(range(circ.num_qudits)))
        num_u3s_needed -= circ.num_qudits
        if num_u3s_needed > 0:
            for cycle, op in circ.operations_with_cycles():
                if isinstance(op.gate, CNOTGate):
                    op_pt = CircuitPoint(cycle, op.location[0])
                    circ.replace_with_circuit(op_pt, flooded_cnot.copy(), as_circuit_gate=True)
                    num_u3s_needed -= 2
                    if num_u3s_needed <= 0:
                        break
            circ.unfold_all()
            ToU3Pass.run_group_circ(circ)
        new_circ_str = lang.encode(circ)
        del circ
        del flooded_cnot
        del init_u3_circ
        return new_circ_str

    @staticmethod
    async def get_final_circ(circ_str: str, do_flood_circ: bool, success_threshold: float, count_t: bool) -> str:
        """Get the final circuit to be used for jiggling"""
        # assert than each circ has U3 gates after their CNOTs
        if do_flood_circ:
            return JiggleEnsemblePass.flood_circ(circ_str)
        
        # Note that do_flood_circ and count_t are mutually exclusive
        assert not (do_flood_circ and count_t)

        num_params = count_params_str(circ_str)
        if num_params == 0:
            return circ_str
        int_thresh = ceil(-1 * np.log10(success_threshold / num_params))
        # empty_circ = Circuit(1)
        if count_t:
            if circ_str.count("gg(") or circ_str.count("gg (") > 0:
                # Already modified
                return circ_str
            circ = lang.decode(circ_str)
            # Replace all RZ gates with GridSynthGate if possible
            pts_to_remove = []
            for cycle, op in circ.operations_with_cycles():
                if isinstance(op.gate, RZGate):
                    gg_params = [op.params[0], min(MIN_EPSILON, int_thresh * 2), 0]
                    circ.replace_gate(CircuitPoint(cycle, op.location[0]), 
                                      GridSynthGate(), op.location, gg_params)
                elif isinstance(op.gate, U3Gate):
                    # if all the params are 0, then remove the gate
                    # print("U3 Gate with params: ", op.params, flush=True)
                    if np.allclose(op.params, [0, 0, 0]):
                        pts_to_remove.append(CircuitPoint(cycle, op.location[0]))
                elif isinstance(op.gate, IdentityGate):
                    pts_to_remove.append(CircuitPoint(cycle, op.location[0]))
            if len(pts_to_remove) > 0:
                circ.batch_pop(pts_to_remove)
                            
            new_circ_str = lang.encode(circ)
            del circ
            return new_circ_str
        else:
            return circ_str

    async def run_ensemble(self, circuit: Circuit, data: PassData) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""
        _logger.debug('Converting single-qubit general gates to U3Gates.')

        if circuit.num_params == 0 and self.count_t:
            return

        # Collected one solution from synthesis

        checkpoint_dir = data["checkpoint_dir"]
        print("Starting JIGGLE ENSEMBLE", flush=True)
        ensemble_file_name = os.path.join(checkpoint_dir, "ensemble_{ind}_{extra}.qasms")
        jiggle_file_name = os.path.join(checkpoint_dir, "ensemble_{ind}_jiggles_{extra}.npy")
        
        start_ens_ind = 0
        jiggle_file = jiggle_file_name.format(ind=start_ens_ind, extra=self.checkpoint_extra_str)

        ens_file = ensemble_file_name.format(ind=0, 
                                             extra=self.checkpoint_extra_str)
        
        # Calculate the number of ensembles from the previous pass
        NUM_ENSEMBLES = 0
        while os.path.exists(ens_file):
            NUM_ENSEMBLES += 1
            ens_file = ensemble_file_name.format(ind=NUM_ENSEMBLES, 
                                             extra=self.checkpoint_extra_str)
            
        # Calculate how many jiggles have been done already
        if os.path.exists(jiggle_file):
            # Check if ensemble has been loaded
            while os.path.exists(jiggle_file):
                start_ens_ind += 1
                jiggle_file = jiggle_file_name.format(ind=start_ens_ind, 
                                                      extra=self.checkpoint_extra_str)

        if start_ens_ind >= NUM_ENSEMBLES:
            return
        
        # Jiggle the rest of the ensembles

        if self.use_calculated_error:
            success_threshold = self.success_threshold * data.get("error_percentage_allocated", 1)
        else:
            success_threshold = self.success_threshold

        for ens_ind in range(start_ens_ind, NUM_ENSEMBLES):
            ens_file = ensemble_file_name.format(ind=ens_ind, 
                                                 extra=self.checkpoint_extra_str)
            circuit_strs = load_ensemble_strs(ens_file)

            if len(circuit_strs) == 0:
                continue
            
            circuit_strs = await get_runtime().map(JiggleEnsemblePass.get_final_circ, 
                                               circuit_strs,
                                               do_flood_circ=self.do_flood_circ,
                                               success_threshold=success_threshold,
                                               count_t=self.count_t)

            '''Get a list of params and caches for each circuit'''
            all_params_caches = await get_runtime().map(JiggleEnsemblePass.single_jiggle_ham, 
                                          circuit_strs, 
                                          target=data.target,
                                          num = self.num_circs, 
                                          success_threshold=success_threshold)
            all_params = [p[0] for p in all_params_caches]
            all_caches = [p[1] for p in all_params_caches]

            ens_file = ensemble_file_name.format(ind=ens_ind, extra=self.checkpoint_extra_str)
            store_ensemble_strs(circuit_strs, ens_file)
            jiggle_file = jiggle_file_name.format(ind=ens_ind, extra=self.checkpoint_extra_str)
            store_params(all_params, jiggle_file)
            cache_file = os.path.join(checkpoint_dir, f"ensemble_{ens_ind}_cache_{self.checkpoint_extra_str}.pkl")
            # store_caches(all_caches, jiggle_file)
            pickle.dump(all_caches, open(cache_file, "wb"))

    async def run_scan_sols(self, circuit: Circuit, data: PassData) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""
        _logger.debug('Converting single-qubit general gates to U3Gates.')

        if circuit.num_params == 0 and self.count_t:
            print("No Params in Circuit, skipping Jiggle Ensemble Pass", flush=True)
            return

        checkpoint_dir = data["checkpoint_dir"]
        ensemble_file_name = os.path.join(checkpoint_dir, "ensemble_{ind}_{extra}.qasms")
        jiggle_file_name = os.path.join(checkpoint_dir, "ensemble_{ind}_jiggles_{extra}.npy")
        probs_file_name = os.path.join(checkpoint_dir, "ensemble_{ind}_probs_{extra}.npy")
        cache_file_name = os.path.join(checkpoint_dir, "ensemble_{ind}_cache_{extra}.pkl")


        # Ensemble 0 is only circuits that epsilon < err_thresh
        ens_file = ensemble_file_name.format(ind=0, extra=self.checkpoint_extra_str)
        jiggle_file = jiggle_file_name.format(ind=0, extra=self.checkpoint_extra_str)
        probs_file = probs_file_name.format(ind=0, extra=self.checkpoint_extra_str)
        cache_file = cache_file_name.format(ind=0, extra=self.checkpoint_extra_str)


        # Ensemble 1 is all circuits with fewer CNOTs
        ens_file_1 = ensemble_file_name.format(ind=1, extra=self.checkpoint_extra_str)
        jiggle_file_1 = jiggle_file_name.format(ind=1, extra=self.checkpoint_extra_str)
        probs_file_1 = probs_file_name.format(ind=1, extra=self.checkpoint_extra_str)
        cache_file_1 = cache_file_name.format(ind=1, extra=self.checkpoint_extra_str)

        # Ensemble 2 is all circuits
        ens_file_2 = ensemble_file_name.format(ind=2, extra=self.checkpoint_extra_str)
        jiggle_file_2 = jiggle_file_name.format(ind=2, extra=self.checkpoint_extra_str)
        probs_file_2 = probs_file_name.format(ind=2, extra=self.checkpoint_extra_str)
        cache_file_2 = cache_file_name.format(ind=2, extra=self.checkpoint_extra_str)

        # if os.path.exists(jiggle_file) and os.path.exists(jiggle_file_1) and os.path.exists(jiggle_file_2):
        #     return
                
        Path(ens_file).parent.mkdir(parents=True, exist_ok=True)
        Path(jiggle_file).parent.mkdir(parents=True, exist_ok=True)
        
        if "ntro_scan_sols" in data:
            scan_sols = data.get("ntro_scan_sols", [])
        else:
            scan_sols = data.get("scan_sols", [])

        circuits: list[Circuit] = [c for c, _ in scan_sols]
        distances: list[float] = [d for _, d in scan_sols]
        if len(circuits) == 0:
            print("No circuits to jiggle, skipping Jiggle Ensemble Pass", flush=True)
            return
        
        eps_circ_inds = []
        for i, d in enumerate(distances):
            if d < self.success_threshold and d > (self.success_threshold ** 2):
                eps_circ_inds.append(i)

        
        less_cnot_circs_inds = []
        orig_count = circuit.count(CNOTGate())
        for i, c in enumerate(circuits):
            cnot_count = c.count(CNOTGate())
            if cnot_count < orig_count:
                less_cnot_circs_inds.append(i)

        print("Eps Circ Inds: ", eps_circ_inds, flush=True)
        print("Less CNOT Circ Inds: ", less_cnot_circs_inds, flush=True)

        success_threshold = self.success_threshold

        circuit_strs = [lang.encode(c) for c in circuits]
        circuit_strs = await get_runtime().map(JiggleEnsemblePass.get_final_circ, 
                                            circuit_strs,
                                            do_flood_circ=self.do_flood_circ,
                                            success_threshold=success_threshold,
                                            count_t=self.count_t)
        
        if self.count_t:
            max_gg = max([c.count("gg") for c in circuit_strs])
            max_gg = min(max_gg, MAX_GGS_TO_JIGGLE)
            max_circs = 4 ** (max_gg)

            all_gg_params = JiggleEnsemblePass.get_all_gg_params(circuit_strs,
                                                                target=data.target,
                                                                success_threshold=self.success_threshold)

            assert len(all_gg_params) == len(circuit_strs)
            all_caches = [p[2] for p in all_gg_params]
            circ_data = [(circuit_strs[i], p[0], p[1]) for i,p in enumerate(all_gg_params)]

            # list of (params, probs) for each circuit
            final_param_probs = await get_runtime().map(
                JiggleEnsemblePass.single_jiggle_ham_clifft,
                circ_data,
                num=max_circs
            )
            all_params = [p[0] for p in final_param_probs]
            all_probs = [p[1] for p in final_param_probs]
            pickle.dump(all_caches, open(cache_file, "wb"))
        else:
            all_params = await get_runtime().map(JiggleEnsemblePass.single_jiggle_ham, 
                                            circuit_strs, 
                                            target=data.target,
                                            num =self.num_circs, 
                                            success_threshold=success_threshold)
            
            # Remove all circuit strs that failed
            bad_inds = [i for i, p in enumerate(all_params) if p is None]
            for ind in reversed(bad_inds):
                del circuit_strs[ind]
                del all_params[ind]
                if ind in less_cnot_circs_inds:
                    less_cnot_circs_inds.remove(ind)
                if ind in eps_circ_inds:
                    eps_circ_inds.remove(ind)

            all_probs = [np.ones((p.shape[0], )) / p.shape[0] for p in all_params]
        
        # Now, only choose circuits with epsilon < err_thresh**2 for ensemble 0
        circuit_strs_eps = [circuit_strs[i] for i in eps_circ_inds]
        all_params_eps = [all_params[i] for i in eps_circ_inds]
        all_probs_eps = [all_probs[i] for i in eps_circ_inds]
        if len(eps_circ_inds) > 0:
            # Store ensemble 0
            if len(all_params_eps[0]) > 0:
                store_params(all_params_eps, jiggle_file)
            store_ensemble_strs(circuit_strs_eps, ens_file)
            store_probs(all_probs_eps, probs_file)

        # Now, only choose circuits with fewer CNOTs for ensemble 1
        circuit_strs_1 = [circuit_strs[i] for i in less_cnot_circs_inds]
        all_params_1 = [all_params[i] for i in less_cnot_circs_inds]
        all_probs_1 = [all_probs[i] for i in less_cnot_circs_inds]

        if len(less_cnot_circs_inds) > 0:
            # Store ensemble 1
            if len(all_params_1[0]) > 0:
                store_params(all_params_1, jiggle_file_1)
            store_ensemble_strs(circuit_strs_1, ens_file_1)
            store_probs(all_probs_1, probs_file_1)

        # Store full ensemble as ensemble 2
        if len(all_params[0]) > 0:
            store_params(all_params, jiggle_file_2)
        store_ensemble_strs(circuit_strs, ens_file_2)
        store_probs(all_probs, probs_file_2)

    async def run(self, circuit: Circuit, data: PassData) -> None:
        if self.use_ensemble:
            await self.run_ensemble(circuit, data)
        elif self.use_scan_sols:
            # Run the jiggle pass on the circuit
            await self.run_scan_sols(circuit, data)
        else:
            pass