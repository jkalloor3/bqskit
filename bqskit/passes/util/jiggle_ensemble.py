"""This module implements the ToU3Pass."""
from __future__ import annotations

import logging

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit
from bqskit.runtime import get_runtime
from typing import Any
from bqskit.ir.opt.cost.functions import HSCostGenerator, FrobeniusCostGenerator
from bqskit.ir.opt.minimizers.lbfgs import LBFGSMinimizer
from bqskit.ir.opt.cost.generator import CostFunctionGenerator
from bqskit.ir.gates import U3Gate, CNOTGate
from bqskit.qis import UnitaryMatrix
import numpy as np
from math import ceil
from itertools import chain
import pickle 
import csv

_logger = logging.getLogger(__name__)


frob_cost = FrobeniusCostGenerator()

def bias_cost(utry: UnitaryMatrix, target: UnitaryMatrix):
    '''
    Calculates the normalized Frobenius distance between two unitaries
    '''
    diff = utry- target
    # This is Frob(u - v)
    cost = np.sqrt(np.real(np.trace(diff @ diff.conj().T)))

    N = utry.shape[0]
    cost = cost / np.sqrt(2 * N)

    # This quantity should be less than HS distance as defined by 
    # Quest Paper 
    return cost

def frobenius_cost(utry: UnitaryMatrix, target: UnitaryMatrix):
    '''
    Calculates the Frobenius distance between two unitaries
    '''
    diff = utry- target
    # This is Frob(u - v)
    cost = np.sqrt(np.real(np.trace(diff @ diff.conj().T)))

    return cost

class  JiggleEnsemblePass(BasePass):
    """Converts single-qubit general unitary gates to U3 Gates."""
    num_jiggles = 0

    def __init__(self, success_threshold = 1e-4, 
                 num_circs = 1000,
                 cost: CostFunctionGenerator = HSCostGenerator(),
                 use_ensemble: bool = True,
                 use_calculated_error: bool = True) -> None:
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
        self.instantiate_options: dict[str, Any] = {
            'dist_tol': self.success_threshold,
            'min_iters': 100,
            'cost_fn_gen': self.cost,
            'method': 'minimization',
            'minimizer': LBFGSMinimizer(),
        }
        self.use_calculated_error = use_calculated_error

    async def get_circ(params: list[float], circuit: Circuit):
        circ_copy = circuit.copy()
        circ_copy.set_params(params)
        return circ_copy

    async def single_jiggle(self, params: list[float], circ: Circuit, dist: float, target: UnitaryMatrix, num: int) -> list[tuple[Circuit, float]]:
        # print("Starting Jiggle", flush=True)
        # start = time.time()
        circs = []
        for _ in range(num):
            cost_fn = self.cost.gen_cost(circ.copy(), target)

            trials = 0

            best_params = np.array(params.copy(), dtype=np.float64)

            extra_diff = max(self.success_threshold - dist, self.success_threshold / len(params))

            while trials < 20:
                trial_costs = []
                if len(params) < 10:
                    num_params_to_jiggle = ceil(len(params) / 2)
                else:
                    num_params_to_jiggle = int(np.random.uniform() * len(params) / 2) + ceil(len(params) / 10)
                # num_params_to_jiggle = len(params)
                params_to_jiggle = np.random.choice(list(range(len(params))), num_params_to_jiggle, replace=False)

                jiggle_amounts = np.random.uniform(-1 * extra_diff, extra_diff, num_params_to_jiggle)
                # print("jiggle_amounts", jiggle_amounts, flush=True)
                next_params = best_params.copy()
                next_params[params_to_jiggle] = next_params[params_to_jiggle] + jiggle_amounts
                circ_cost = cost_fn.get_cost(next_params)
                trial_costs.append(circ_cost)
                if (circ_cost < self.success_threshold):
                    extra_diff = extra_diff * 1.5
                    best_params = next_params
                    # Randomly choose to finish early
                    if np.random.uniform() < 0.2 and trials > 4:
                        break
                else:
                    extra_diff = extra_diff / 10
                trials += 1
            
            circ_copy = circ.copy()
            circ_copy.set_params(best_params)
            # circ_cost = cost_fn.get_cost(best_params)
            circ_cost = frob_cost.calc_cost(circ_copy, target)
            circs.append((circ_copy, circ_cost))
        JiggleEnsemblePass.num_jiggles += 1
        return circs


    async def jiggle_circ(self, circ_dist: tuple[Circuit, float], target: UnitaryMatrix, num_circs: int) -> list[tuple[Circuit, float]]:
        circ, dist = circ_dist
        
        params = circ.params

        if len(params) < 10:
            # print(circ.gate_counts)
            # print("WTF NO PARAMS")
            for i in range(circ.num_qudits):
                # Add a layer of U3 gates
                circ.append_gate(U3Gate(), (i,), [0, 0, 0])
            params = circ.params
        all_circs = []

        if num_circs > 60:
            # print(f"Awaiting All {num_circs // 50} Jiggles")
            print(f"Launching {ceil(num_circs / 20)} Tasks", flush=True)
            all_circs: list[list[tuple[Circuit, float]]] = await get_runtime().map(self.single_jiggle, [params] * ceil(num_circs / 20), circ=circ, dist=dist, target=target, num=20)
            print(f"Finished {ceil(num_circs / 20)} Tasks", flush=True)
            all_circs: list[tuple[Circuit, float]] = list(chain.from_iterable(all_circs))
        else:
            all_circs: list[tuple[Circuit, float]] = await self.single_jiggle(params, circ, dist, target, num_circs)
        
        return all_circs


    def subselect_circs_by_bias(self, unitaries: list[UnitaryMatrix], target: UnitaryMatrix) -> tuple[list[int], float]:
        '''
        Given a list of circuits and distances, subselect the circuits
        that minimize the bias between the target and the mean unitary
        of the ensemble.


        The circuits are sorted by the gate reduction, so try to use
        the circuits that have the fewest gates to minimize the bias.
        '''
        # Initially randomly sort circs
        order = np.random.permutation(len(unitaries))

        # Grab first 25 as initial set
        inds = list(order[:25])
        avg_un = np.mean([unitaries[i] for i in inds], axis=0)
        bias = bias_cost(avg_un, target)
        
        cur_ind = 25
        nn = 1
        while cur_ind < len(unitaries) and len(inds) < 1000:
            num_to_check = min(nn, len(unitaries) - cur_ind)
            # Get next unitaries to try and add
            next_inds = [order[cur_ind + i] for i in range(num_to_check)]
            next_uns = [unitaries[i] for i in next_inds]

            # Try to add all of them
            num_inds = len(inds)
            new_avg_un = (num_inds / (num_inds + num_to_check)) * avg_un + (num_to_check / (num_inds + num_to_check)) * np.mean(next_uns, axis=0)
            new_bias = bias_cost(new_avg_un, target)

            # If the bias is less, add the unitaries
            if new_bias < bias:
                print("Taking all together")
                print("Old Bias", bias)
                print("New Bias", new_bias)
                bias = new_bias
                avg_un = new_avg_un
                inds.extend(next_inds)
                # nn += 5
            # else:
            #     nn -= 3
            #     nn = max(5, nn)

            cur_ind = cur_ind + num_to_check

        # Return the ensemble with the best bias
        return inds, bias


    async def run(self, circuit: Circuit, data: PassData) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""
        _logger.debug('Converting single-qubit general gates to U3Gates.')

        # Collected one solution from synthesis
        print("Starting JIGGLE ENSEMBLE", flush=True)
        
        if "finished_jiggle" in data:
            print("Already Jiggled", flush=True)
            return

        print("Number of ensembles", len(data["ensemble"]), flush=True)

        if self.use_calculated_error:
            # print("OLD", self.success_threshold)
            self.success_threshold = self.success_threshold * data["error_percentage_allocated"]
            # print("NEW", self.success_threshold)

        data["finished_jiggle"] = False

        ensemble = []
        futures = []

        for scan_sols in data["ensemble"]:
            all_circs = []
            # print("Number of SCAN SOLS", len(scan_sols))

            # For each params come up with nth root of num_circs number of extra params
            if self.use_ensemble:
                circuits = [psol[0] for psol in scan_sols]
                dists = [psol[1] for psol in scan_sols]
            else:
                circuits = [circuit]
                dists = [self.cost.calc_cost(circuit, data.target)]

            circ_dists = list(zip(circuits, dists))
            
            print(f"Launching {len(circ_dists)} Tasks", flush=True)
            all_circs_future = get_runtime().map(self.jiggle_circ,
                                                circ_dists,
                                                target=data.target,
                                                num_circs = ceil(self.num_circs / len(circuits)))
            futures.append(all_circs_future)

        print("Awaiting All Jiggles", flush=True)
        all_circs = [await future for future in futures]
        ensemble: list[list[tuple[Circuit, float]]] = [list(chain.from_iterable(all_circ)) for all_circ in all_circs]

        print("Number of Circs post Jiggle", [len(ens) for ens in ensemble])

        # Now subselect based on bias
        print("Subselecting Circuits", flush=True)
        ensemble_names = ["Random Sub-Sample", "Least CNOTs", "Medium CNOTs", "Valid CNOTs"]
        target = data.target
        csv_dict = []
        for i,ens in enumerate(ensemble):
            ensemble_data = {}
            unitaries: list[UnitaryMatrix] = [x[0].get_unitary() for x in ens]
            e1s = [bias_cost(un, target) for un in unitaries]
            e1s_actual = [frobenius_cost(un, target) for un in unitaries]
            e1 = np.mean(e1s)
            e1_actual = np.mean(e1s_actual)
            mean_un = np.mean(unitaries, axis=0)
            orig_bias = bias_cost(mean_un, target)
            orig_bias_actual = frobenius_cost(mean_un, target)
            
            final_counts = [circ.count(CNOTGate()) for circ, _ in ens]

            ensemble_data["Ensemble Generation Method"] = ensemble_names[i]
            ensemble_data["Epsilon"] = e1
            ensemble_data["Epsilon Actual"] = e1_actual
            ensemble_data["Avg CNOT Count before subselect"] = np.mean(final_counts)
            ensemble_data["Bias without subselect"] = orig_bias
            ensemble_data["Bias without subselect Actual"] = orig_bias_actual
            ensemble_data["Num Circs"] = len(ens)
            subselected_inds, new_bias = self.subselect_circs_by_bias(unitaries, target)
            new_mean_un = np.mean([unitaries[i] for i in subselected_inds], axis=0)
            new_bias_actual = frobenius_cost(new_mean_un, target)

            subselected_circs_dists = [ens[i]for i in subselected_inds]
            final_counts = [circ.count(CNOTGate()) for circ, _ in subselected_circs_dists]

            ensemble_data["Avg CNOT Count after subselect"] = np.mean(final_counts)
            ensemble_data["Num Circs after subselect"] = len(subselected_circs_dists)
            ensemble_data["Bias after subselect"] = new_bias
            ensemble_data["Bias after subselect Actual"] = new_bias_actual

            csv_dict.append(ensemble_data)


        data["ensemble"] = ensemble

        if "checkpoint_dir" in data:
            checkpoint_data_file = data["checkpoint_data_file"]
            csv_file = checkpoint_data_file.replace(".data", ".csv_subselect2")
            writer = csv.DictWriter(open(csv_file, "w", newline=""), fieldnames=csv_dict[0].keys())
            writer.writeheader()
            for row in csv_dict:
                writer.writerow(row)

        # if "checkpoint_dir" in data:
        #     data["finished_jiggle"] = True
        #     checkpoint_data_file = data["checkpoint_data_file"]
        #     pickle.dump(data, open(checkpoint_data_file, "wb"))


        return

        
