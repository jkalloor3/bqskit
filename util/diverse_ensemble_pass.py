"""This module implements the ToU3Pass."""
from __future__ import annotations

import logging

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit
from bqskit.runtime import get_runtime
from typing import Any
from bqskit.ir.lang.qasm2 import OPENQASM2Language
from bqskit.ir.opt.cost.functions import GPNormalizedFrobeniusCostGenerator, GPNormalizedFrobeniusCostGenerator
from bqskit.ir.opt.minimizers.lbfgs import LBFGSMinimizer
from bqskit.ir.opt.cost.generator import CostFunctionGenerator
import pickle
from bqskit.ir import Circuit
from bqskit.runtime import get_runtime

import os
from .common import (store_params,store_ensemble_strs, store_probs)
from .jiggle_ensemble import JiggleEnsemblePass, MAX_GGS_TO_JIGGLE
from .gg import gg_gate_def
from .fix_angles import FixAnglesPass
from .convert_to_cliff import ConvertToZXZXZSimple

import numpy as np
from math import ceil

_logger = logging.getLogger(__name__)

frob_cost = GPNormalizedFrobeniusCostGenerator()
lang = OPENQASM2Language(gate_defs=[("gg", gg_gate_def)])

class DefaultGGEnsemblePass(BasePass):
    """Converts single-qubit general unitary gates to U3 Gates."""
    num_jiggles = 0
    finished_pass_str = "finished_jiggle"

    def __init__(self, 
                 success_threshold = 1e-4, 
                 cost: CostFunctionGenerator = GPNormalizedFrobeniusCostGenerator(),
                 checkpoint_extra_str: str = "") -> None:
        """
        Construct a ToU3Pass.

        Args:
            convert_all_single_qubit_gates (bool): Indicates wheter to convert
            only the general gates, or every single qubit gate.
        """

        self.success_threshold = success_threshold
        self.cost = cost
        self.instantiate_options: dict[str, Any] = {
            'dist_tol': self.success_threshold,
            'min_iters': 100,
            'cost_fn_gen': self.cost,
            'method': 'minimization',
            'minimizer': LBFGSMinimizer(),
        }
        self.checkpoint_extra_str = checkpoint_extra_str

    def fix_circuits(self, circuits: list[Circuit], eps: float) -> None:
        '''
        Fix all GG gate angles in the circuits to within eps distance of target.
        '''

        for circ in circuits:
            # Fix angles to high precision
            FixAnglesPass.run_circ(circ, 15)
            # Convert to ZXZXZ simple
            ConvertToZXZXZSimple.run_circuit(circ)
            # Fix angles again to high precision
            FixAnglesPass.run_circ(circ, 15)
            # Calculate the precision based on remaining params
            num_params = circ.num_params
            error_per_param = eps / num_params
            precision = ceil(-np.log10(error_per_param))
            print(circ.gate_counts)
            print(f"Precision: {precision}, Num Params: {num_params}", flush=True)

            # Now fix angles to desired precision
            FixAnglesPass.run_circ(circ, precision)

    
    
    async def run_circuit(self, 
                          circuit: Circuit, 
                          file_names: tuple,
                          target: np.ndarray) -> None:
        success_threshold = self.success_threshold
        # First, fix angles to distance eps^2
        self.fix_circuits([circuit], success_threshold ** 2)

        # Print distances
        dists = [frob_cost.calc_cost(circ, target) for circ in [circuit]]
        print("Distances after angle fixing: ", dists, flush=True)

        # Now jiggle the circuits
        circuit_strs = [lang.encode(c) for c in [circuit]]
        circuit_strs = await get_runtime().map(JiggleEnsemblePass.get_final_circ,
                                            circuit_strs,
                                            do_flood_circ=False,
                                            success_threshold=success_threshold,
                                            count_t=True)
        
        max_gg = max([c.count("gg") for c in circuit_strs])
        max_gg = min(max_gg, MAX_GGS_TO_JIGGLE)
        max_circs = 4 ** (max_gg)

        all_gg_params = JiggleEnsemblePass.get_all_gg_params(circuit_strs,
                                                            target=target,
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

        ens_file, jiggle_file, probs_file, cache_file = file_names

        # Store default ensemble
        def_params = [p[0] for p in final_param_probs]
        def_probs = [p[1] for p in final_param_probs]
        store_ensemble_strs(circuit_strs, ens_file)
        store_params(def_params, jiggle_file)
        store_probs(def_probs, probs_file)
        pickle.dump(all_caches, open(cache_file, "wb"))
    
    
    async def run(self, circuit: Circuit, data: PassData) -> None:
        '''
        Generate an ensemble over the un-synthesized original circuit.
        
        Process:
        1. Fix angles to distance eps^2
        2. Jiggle GG gates to distance eps
        3. Output ensemble circuits and jiggles to files
        '''

        checkpoint_dir = data["checkpoint_dir"]
        ensemble_file_name = os.path.join(checkpoint_dir, "ensemble_{ind}_{extra}.qasms")
        jiggle_file_name = os.path.join(checkpoint_dir, "ensemble_{ind}_jiggles_{extra}.npy")
        probs_file_name = os.path.join(checkpoint_dir, "ensemble_{ind}_probs_{extra}.npy")
        cache_file_name = os.path.join(checkpoint_dir, "ensemble_{ind}_cache_{extra}.pkl")
        ens_file = ensemble_file_name.format(ind="def", extra=self.checkpoint_extra_str)
        jiggle_file = jiggle_file_name.format(ind="def", extra=self.checkpoint_extra_str)
        probs_file = probs_file_name.format(ind="def", extra=self.checkpoint_extra_str)
        cache_file = cache_file_name.format(ind="def", extra=self.checkpoint_extra_str)

        ens_file_ntro = ensemble_file_name.format(ind="ntro", extra=self.checkpoint_extra_str)
        jiggle_file_ntro = jiggle_file_name.format(ind="ntro", extra=self.checkpoint_extra_str)
        probs_file_ntro = probs_file_name.format(ind="ntro", extra=self.checkpoint_extra_str)
        cache_file_ntro = cache_file_name.format(ind="ntro", extra=self.checkpoint_extra_str)

        ens_file_0 = ensemble_file_name.format(ind="0", extra=self.checkpoint_extra_str)

        if not os.path.exists(ens_file_0):
            print("Base ensemble not done, skipping Default Ensemble", flush=True)
            return

        circuits = []
        file_names = []
        if os.path.exists(cache_file):
            # Just do NTRO ensemble
            circuits += [data["e2_scan_sols"][0][0].copy()]
            file_names += [(ens_file_ntro, jiggle_file_ntro, 
                           probs_file_ntro, cache_file_ntro)]
        elif os.path.exists(cache_file_ntro):
            print("NTRO ensemble exists, loading circuits", flush=True)
            return
        else:
            # Do both default and NTRO ensembles
            circuits += [circuit.copy()]
            file_names = [(ens_file, jiggle_file, 
                           probs_file, cache_file)]
            

        for circ, fnames in zip(circuits, file_names):
            await self.run_circuit(circ, fnames, data.target)
