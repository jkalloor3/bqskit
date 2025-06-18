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
import itertools
from bqskit.ir.lang.qasm2 import OPENQASM2Language
from bqskit.ir.opt.cost.functions import GPNormalizedFrobeniusCostGenerator, GPNormalizedFrobeniusCostGenerator
from bqskit.ir.opt.minimizers.lbfgs import LBFGSMinimizer
from bqskit.ir.opt.cost.generator import CostFunctionGenerator
from bqskit.ir.gates import *
from bqskit.qis import UnitaryMatrix
import numpy as np
from math import ceil
import pickle
import time

from bqskit.qis.pauli import PauliMatrices
from bqskit.ir import Circuit
from bqskit.qis import UnitaryMatrix
import scipy as sp
from bqskit.utils.math import dot_product
from bqskit.runtime import get_runtime

import os
from .common import (store_params, load_ensemble_strs, store_ensemble_strs, 
                     calc_num_circs)
from .counter import count_params_str
from .gg import GridSynthGate, gg_gate_def, MIN_EPSILON, get_rz_perturbations

_logger = logging.getLogger(__name__)

frob_cost = GPNormalizedFrobeniusCostGenerator()
lang = OPENQASM2Language(gate_defs=[("gg", gg_gate_def)])
class  JiggleEnsemblePass(BasePass):
    """Converts single-qubit general unitary gates to U3 Gates."""
    num_jiggles = 0

    finished_pass_str = "finished_jiggle"

    def __init__(self, success_threshold = 1e-4, 
                 num_circs = 1000,
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
        del final_matrices
        del perturbations
        del u3_utry
        return final_params

    @staticmethod
    def single_jiggle_ham(circ_str: str, target: UnitaryMatrix, 
                          num: int, success_threshold: float) -> np.ndarray[float]:
        '''
        Apply Hamiltonian perturbations to the circuit.

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

        Also returns the local cache for Gridsynth Strings
        
        '''
        circ = lang.decode(circ_str)
        dist = frob_cost.calc_cost(circ, target)
        # For each U3 gate, calculate do a Hamiltonian perturbation
        num_u3s = circ.count(U3Gate())
        num_ggs = circ.count(GridSynthGate())
        num_rzs = circ.count(RZGate())
        # print("Init Dist: ", dist, flush=True)
        if (num_u3s + num_rzs + num_ggs) == 0:
            print("No U3s or Zs", circ.gate_counts, flush=True)
            # Return a num x 1 array of zeros
            empty_params = np.zeros((num * 2, circ.num_params))
            return empty_params, None
        
        if num_ggs > 0:
            local_cache = {}
        else:
            local_cache = None

        # For each u3, come up with 16 param perturbations
        num_options = 16
        # Map U3 params to perturbed params
        u3_param_options: dict[tuple[float, float, float], 
                               list[list[float]]] = {}
        
        # Map RZ angle to potential perturbed angles
        rz_param_options: dict[float, list[float]] = {}
        # Map GG angles to potential GG params and probabilities
        gg_param_options: dict[float, list[list[float]]] = {}
        gg_probs: dict[float, list[float]] = {}

        orig_perturb_dist = success_threshold - dist
        perturb_dist = orig_perturb_dist / (num_u3s + num_rzs + num_ggs + 1)
        # Round log of dist to nearest int
        # Add 2 because epsilon != frob_cost
        try:
            int_perturb_dist = ceil(-1 * np.log10(perturb_dist))
        except:
            print("Success Threshold: ", success_threshold, flush=True)
            print("Dist: ", dist, flush=True)
            print("Perturb Dist is 0", flush=True)
            return np.vstack([circ.params] * num * 2), local_cache
        # start_time = time.process_time()
        for op in circ.operations():
            if isinstance(op.gate, U3Gate):
                # Round params to int_perturb_dist
                key = tuple(np.round(op.params, int_perturb_dist + 1))
                if key not in u3_param_options:
                    u3_param_options[key] = JiggleEnsemblePass.get_ham_perturbations(op.get_unitary(), perturb_dist, num_options * 2)
            if isinstance(op.gate, RZGate):
                angle = op.params[0]
                if angle not in rz_param_options:
                    z_perturbs = np.array([np.pi/ 2, -np.pi/2, np.pi, -np.pi]) * perturb_dist
                    rz_param_options[angle] = z_perturbs + angle
            elif isinstance(op.gate, GridSynthGate):
                angle = op.params[0]
                if angle not in gg_param_options:
                    gg_params, gg_strs, probs = get_rz_perturbations(angle, 
                                                                     int_perturb_dist)
                    
                    # Fill up local cache with gg_strs
                    for i, gg_str in enumerate(gg_strs):
                        ind = (gg_params[i][0], gg_params[i][1])
                        local_cache[ind] = gg_str

                    gg_param_options[angle] = gg_params
                    gg_probs[angle] = probs

        final_params = []
        new_circ = circ.copy()
        for _ in range(num):
            new_circ.set_params(circ.params)
            u3_ind = 0
            gg_ind = 0
            z_ind = 0
            rand_u3_inds = np.random.choice(num_options, num_u3s, replace=True)
            rand_rz_inds = np.random.choice(2, num_rzs, replace=True)
            for op in new_circ.operations():
                if isinstance(op.gate, U3Gate):
                    key = tuple(np.round(op.params, int_perturb_dist + 1))
                    op.params = u3_param_options[key][rand_u3_inds[u3_ind] * 2]
                    u3_ind += 1
                if isinstance(op.gate, GridSynthGate):
                    param_options = gg_param_options[op.params[0]]
                    probs = gg_probs[op.params[0]]
                    rand_ind = np.random.choice(len(param_options), p=probs)
                    op.params = param_options[rand_ind]
                    gg_ind += 1
                if isinstance(op.gate, RZGate):
                    op.params = [rz_param_options[op.params[0]][rand_rz_inds[z_ind] * 2]]
                    z_ind += 1

            full_params_1 = new_circ.params
            u3_ind = 0
            gg_ind = 0
            z_ind = 0
            new_circ.set_params(circ.params)
            for op in new_circ.operations():
                if isinstance(op.gate, U3Gate):
                    key = tuple(np.round(op.params, int_perturb_dist + 1))
                    op.params = u3_param_options[key][rand_u3_inds[u3_ind] * 2 + 1]
                    u3_ind += 1
                if isinstance(op.gate, GridSynthGate):
                    param_options = gg_param_options[op.params[0]]
                    probs = gg_probs[op.params[0]]
                    rand_ind = np.random.choice(len(param_options), p=probs)
                    op.params = param_options[rand_ind]
                    gg_ind += 1
                if isinstance(op.gate, RZGate):
                    op.params = [rz_param_options[op.params[0]][rand_rz_inds[z_ind] * 2 + 1]]
                    z_ind += 1
            full_params_2 = new_circ.params
            final_params.append(full_params_1)
            final_params.append(full_params_2)

        return np.vstack(final_params), local_cache

    @staticmethod
    def flood_circ(circ_str: str) -> str:
        circ = lang.decode(circ_str)
        flooded_cnot = Circuit(2)
        flooded_cnot.append_gate(CNOTGate(), (0, 1))
        flooded_cnot.append_gate(U3Gate(), (0,), [0, 0, 0])
        flooded_cnot.append_gate(U3Gate(), (1,), [0, 0, 0])
        init_u3_circ = Circuit(circ.num_qudits)
        for i in range(circ.num_qudits):
            init_u3_circ.append_gate(U3Gate(), (i,), [0, 0, 0])
        circ.insert_circuit(0, init_u3_circ, tuple(range(circ.num_qudits)))
        for cycle, op in circ.operations_with_cycles():
            if isinstance(op.gate, CNOTGate):
                op_pt = CircuitPoint(cycle, op.location[0])
                circ.replace_with_circuit(op_pt, flooded_cnot.copy(), as_circuit_gate=True)
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
        int_thresh = ceil(-1 * np.log10(success_threshold / num_params)) + 1
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
                    gg_params = [op.params[0], min(MIN_EPSILON, int_thresh * 2 + 2), 0]
                    circ.replace_gate(CircuitPoint(cycle, op.location[0]), 
                                      GridSynthGate(), op.location, gg_params)
                elif isinstance(op.gate, U3Gate):
                    # if all the params are 0, then remove the gate
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
        # checkpoint_dir = "/pscratch/sd/j/jkalloor/bqskit/block_checkpoints_final_paper_clifft/QITE_8_1_0_5.0"
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
            print("Finished Jiggle Ensemble Pass", flush=True)
            return
        
        # Jiggle the rest of the ensembles

        if self.use_calculated_error:
            success_threshold = self.success_threshold * data.get("error_percentage_allocated", 1)
        else:
            success_threshold = self.success_threshold

        futs = []
        start = time.time()
        for ens_ind in range(start_ens_ind, NUM_ENSEMBLES):
            ens_file = ensemble_file_name.format(ind=ens_ind, 
                                                 extra=self.checkpoint_extra_str)
            print("Loading Ensemble", ens_file, flush=True)
            load_start = time.time()
            circuit_strs = load_ensemble_strs(ens_file)
            load_time = time.time() - load_start
            print("Number of Circuits", len(circuit_strs), load_time, flush=True)

            if len(circuit_strs) == 0:
                continue
            
            mod_start = time.time()
            circuit_strs = await get_runtime().map(JiggleEnsemblePass.get_final_circ, 
                                               circuit_strs,
                                               do_flood_circ=self.do_flood_circ,
                                               success_threshold=success_threshold,
                                               count_t=self.count_t)
            mod_time = time.time() - mod_start
            print("Modified Circuits", mod_time, flush=True)

            '''Get a list of params and caches for each circuit'''
            all_params_caches = await get_runtime().map(JiggleEnsemblePass.single_jiggle_ham, 
                                          circuit_strs, 
                                          target=data.target,
                                          num = ceil(self.num_circs / len(circuit_strs)), 
                                          success_threshold=success_threshold)
            all_params = [p[0] for p in all_params_caches]
            all_caches = [p[1] for p in all_params_caches]

            ens_file = ensemble_file_name.format(ind=ens_ind, extra=self.checkpoint_extra_str)
            store_start = time.time()
            store_ensemble_strs(circuit_strs, ens_file)
            print("Finished Jiggling Ensemble", flush=True)
            print("Total Bytes", sum(p.nbytes for p in all_params) / 1024 / 1024 / 1024, flush=True)
            del circuit_strs
            store_time = time.time() - store_start
            jiggle_file = jiggle_file_name.format(ind=ens_ind, extra=self.checkpoint_extra_str)
            store_params(all_params, jiggle_file)
            cache_file = os.path.join(checkpoint_dir, f"ensemble_{ens_ind}_cache.pkl")
            # store_caches(all_caches, jiggle_file)
            pickle.dump(all_caches, open(cache_file, "wb"))
            print("Stored Ensemble", store_time, flush=True)

        total_time = time.time() - start
        print("Total Time for Jiggling Params: ", total_time, flush=True)


    async def run_scan_sols(self, circuit: Circuit, data: PassData) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""
        _logger.debug('Converting single-qubit general gates to U3Gates.')

        if circuit.num_params == 0 and self.count_t:
            print("No Params in Circuit, skipping Jiggle Ensemble Pass", flush=True)
            return

        checkpoint_dir = data["checkpoint_dir"]
        print("Checkpoint Dir: ", checkpoint_dir, flush=True)
        print("Starting JIGGLE ENSEMBLE", flush=True)
        # checkpoint_dir = "/pscratch/sd/j/jkalloor/bqskit/block_checkpoints_final_paper_clifft/QITE_8_1_0_5.0"
        ensemble_file_name = os.path.join(checkpoint_dir, "ensemble_{ind}_{extra}.qasms")
        jiggle_file_name = os.path.join(checkpoint_dir, "ensemble_{ind}_jiggles_{extra}.npy")
        ens_file = ensemble_file_name.format(ind=0, extra=self.checkpoint_extra_str)
        jiggle_file = jiggle_file_name.format(ind=0, extra=self.checkpoint_extra_str)
        if os.path.exists(ens_file):
            num_circs = calc_num_circs(jiggle_file)
            if num_circs > self.num_circs:
                print(f"Already have {num_circs} circuits in ensemble, " \
                "skipping Jiggle Ensemble Pass", flush=True)
                return
                
        Path(ens_file).parent.mkdir(parents=True, exist_ok=True)
        Path(jiggle_file).parent.mkdir(parents=True, exist_ok=True)
        
        scan_sols = data["scan_sols"]
        circuits = [c for c, _ in scan_sols]
        # Jiggle the rest of the ensembles

        if self.use_calculated_error:
            success_threshold = self.success_threshold * data.get("error_percentage_allocated", 1)
        else:
            success_threshold = self.success_threshold

        circuit_strs = [lang.encode(c) for c in circuits]
        mod_start = time.time()
        circuit_strs = await get_runtime().map(JiggleEnsemblePass.get_final_circ, 
                                            circuit_strs,
                                            do_flood_circ=self.do_flood_circ,
                                            success_threshold=success_threshold,
                                            count_t=self.count_t)
        mod_time = time.time() - mod_start
        print("Modified Circuits", mod_time, flush=True)

        '''Get a list of params and caches for each circuit'''
        all_params_caches = await get_runtime().map(JiggleEnsemblePass.single_jiggle_ham, 
                                        circuit_strs, 
                                        target=data.target,
                                        num = ceil(self.num_circs / len(circuit_strs)), 
                                        success_threshold=success_threshold)
        
        all_params = [p[0] for p in all_params_caches]
        all_caches = [p[1] for p in all_params_caches]
        store_start = time.time()
        store_ensemble_strs(circuit_strs, ens_file)
        print("Finished Jiggling Ensemble", flush=True)
        store_time = time.time() - store_start
        if len(all_params[0]) > 0:
            store_params(all_params, jiggle_file)

        if self.count_t:
            cache_file = os.path.join(checkpoint_dir, f"ensemble_0_cache.pkl")
            Path(cache_file).parent.mkdir(parents=True, exist_ok=True)
            pickle.dump(all_caches, open(cache_file, "wb"))
            print("Stored Ensemble", store_time, flush=True)
            if self.pass_ensemble:
                # Calculate ensemble
                data['ensemble_circs'] = circuit_strs
                data['ensemble_jiggles'] = all_params
                data["ensemble_cache"] = all_caches

    async def run(self, circuit: Circuit, data: PassData) -> None:
        if self.use_ensemble:
            await self.run_ensemble(circuit, data)
        elif self.use_scan_sols:
            # Run the jiggle pass on the circuit
            print("Running Jiggle Ensemble Pass on Scan Sols", flush=True)
            await self.run_scan_sols(circuit, data)
        else:
            pass