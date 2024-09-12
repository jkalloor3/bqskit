"""This module implements the ToU3Pass."""
from __future__ import annotations

import logging

from bqskit.compiler.basepass import BasePass
from bqskit.passes import ForEachBlockPass
from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit, CircuitPoint, Operation, CircuitLocationLike
from bqskit.runtime import get_runtime
from typing import Any
from collections import deque
from bqskit.ir.opt.cost.functions import HSCostGenerator, HilbertSchmidtCostGenerator, FrobeniusCostGenerator
from bqskit.ir.opt.minimizers.lbfgs import LBFGSMinimizer
from bqskit.ir.opt.cost.generator import CostFunctionGenerator
from bqskit.ir.gates import CircuitGate
from bqskit.ir.gates import CNOTGate, FixedRZGate
from bqskit.qis import UnitaryMatrix
import numpy as np
import pickle
import random


_logger = logging.getLogger(__name__)

frob_cost = FrobeniusCostGenerator()
class CreateEnsemblePass(BasePass):
    """Converts single-qubit general unitary gates to U3 Gates."""

    def __init__(self, success_threshold = 1e-4, 
                 num_circs = 1000,
                 cost: CostFunctionGenerator = HSCostGenerator(),
                 use_calculated_error: bool = False,
                 num_random_ensembles: int = 3,
                 solve_exact_dists: bool = False,
                 sort_by_t: bool = False) -> None:
        """
        Construct a ToU3Pass.

        Args:
            convert_all_single_qubit_gates (bool): Indicates wheter to convert
            only the general gates, or every single qubit gate.
        """

        self.success_threshold = success_threshold
        self.num_circs = num_circs
        self.cost = cost
        self.solve_exact_dists = solve_exact_dists
        self.use_calculated_error = use_calculated_error
        self.num_random_ensembles = num_random_ensembles
        self.instantiate_options={
            'min_iters': 100,
            'ftol': self.success_threshold,
            'gtol': 1e-10,
            'cost_fn_gen': self.cost,
            'dist_tol': self.success_threshold,
            'method': 'qfactor'
        }
        self.sort_by_t = sort_by_t

    async def unfold_circ(
            config_dist: tuple[list[CircuitGate], float], 
            circuit: Circuit, 
            pts: list[CircuitPoint], 
            locations: list[CircuitLocationLike], 
            solve_exact_dists: bool = False, 
            cost: CostFunctionGenerator = HilbertSchmidtCostGenerator(),
            target: UnitaryMatrix = None) -> tuple[Circuit, float]:
        
        config, dist = config_dist
        operations = [
            Operation(cg, loc, cg._circuit.params)
            for cg, loc
            in zip(config, locations)
        ]
        copied_circuit = circuit.copy()
        copied_circuit.batch_replace(pts, operations)
        copied_circuit.unfold_all()

        if solve_exact_dists:
            dist = cost.calc_cost(copied_circuit, target)

        return copied_circuit, dist


    async def knapsack_solve(self, psol_diffs: list[list[int]], psol_dists: list[list[float]], total_dist: float, num_circs: int, greediest: bool = False) -> list:
        '''
        Pick an ensemble of circuits that minimizes the total number of CNOTs while keeping the total distance below a threshold
        You must pick one psol from each list of psol_diffs.

        

        Args:
            psol_diffs: list of lists of differences in CNOT counts 
            dists: list of lists of distances
            total_dist: total distance allowed

        Returns:
            list of list of indices to pick. Each list corresponds to a list of psols. 
            There are `num_circs` lists in total.
        
        '''

        # Most greedy algorithm
        # Pick the psol that minimizes the difference in CNOT counts
        diffs = np.ones([len(psol_diffs),len(max(psol_diffs,key = lambda x: len(x)))])
        dists = np.zeros_like(diffs)
        for i,j in enumerate(psol_diffs):
            diffs[i][0:len(j)] = j
            dists[i][0:len(j)] = psol_dists[i]

        # Sort by best CNOT count diff
        if greediest:
            orig_inds = np.argsort(diffs[:, 0])
        else:
            # Pick a random ordering
            orig_inds = np.random.choice(len(diffs), size=(len(diffs)), replace=False)
        
        all_valid_inds = self.BFS(orig_inds, psol_diffs, dists, total_dist, greedy=greediest)

        sorted_inds = sorted(all_valid_inds, key=lambda x: x[1])

        # print("SORTED Savings", [x[1] for x in sorted_inds], flush=True)

        if greediest:
            return [(x[0], x[2]) for x in sorted_inds[:num_circs]]
        else:
            weights = np.array([(x[1] - 0.001) for x in sorted_inds])
            weights = weights / np.sum(weights)
            # Add uniform distribution to encourage exploration
            weights += 1/len(weights)
            weights = weights / np.sum(weights)
            try:
                random_inds = np.random.choice(len(sorted_inds), size=(num_circs), p=weights)
            except Exception as e:
                print("Weights", weights, flush=True)
                print("Sorted Inds", sorted_inds, flush=True)
                _logger.error(e)
                
            return [(sorted_inds[i][0], sorted_inds[i][2]) for i in random_inds]
    
    def BFS(self, orig_inds, psol_diffs, dists, total_dist, greedy = False):
        queue = deque([])
        queue.append((orig_inds, 0, [-1 for _ in psol_diffs], 0))
        final_inds = []
        total_sols = 1
        max_total_sols = 10000
        while len(queue) > 0:
            inds, cur_dist, greedy_inds, cnots_saved = queue.popleft()
            if len(inds) == 0:
                final_inds.append((greedy_inds, cnots_saved, cur_dist))
            else:
                ind = inds[0]
                pos_sols = 0

                if greedy:
                    cols = np.arange(len(psol_diffs[ind]))
                else:
                    cols = np.random.choice(len(psol_diffs[ind]), size=(len(psol_diffs[ind])), replace=False)


                for j in cols:
                    dist = dists[ind][j]
                    new_dist = cur_dist + dist
                    if new_dist < total_dist:
                        pos_sols += 1

                new_total_sols = total_sols * pos_sols
                if new_total_sols > max_total_sols:
                    # Only explore one solution
                    pos_sols = 1
                else:
                    pos_sols = min(pos_sols, 3)

                total_sols *= pos_sols
                for j in cols:
                    dist = dists[ind][j]
                    new_dist = cur_dist + dist
                    if new_dist < total_dist:
                        pos_sols -= 1
                        new_greedy_inds = greedy_inds.copy()
                        new_greedy_inds[ind] = j
                        new_cnots_saved = cnots_saved + psol_diffs[ind][j]
                        queue.append((inds[1:], new_dist, new_greedy_inds, new_cnots_saved))
                    if pos_sols == 0:
                        break

        return final_inds

    async def assemble_circuits(
        self,
        circuit: Circuit,
        psols: list[list[Circuit]],
        pts: list[CircuitPoint],
        dists: list[list[float]],
        targets: list[UnitaryMatrix],
        thresholds: list[float],
        target: UnitaryMatrix = None
    ) -> Circuit:
        """Assemble a circuit from a list of block indices."""
        if self.sort_by_t:
            # The fewer parameters the better
            psols_diffs = [[(psol.num_params - psols[i][-1].num_params) for psol in psols[i]] for i in range(len(psols))]
        else:
            # The fewer CNOTs the better
            psols_diffs = [[psol.count(CNOTGate()) - psols[i][-1].count(CNOTGate()) for psol in psols[i]] for i in range(len(psols))]
        locations = [circuit[pt].location for pt in pts]

        possible_sols = 1
        for i in range(len(psols_diffs)):  
            possible_sols *= len(psols_diffs[i])
        
        if possible_sols == 1:
            print("Only one solution", flush=True)
            circ_list = [psols[i][-1] for i in range(len(psols))]
            default_config = [CircuitGate(circ) for circ in circ_list]
            all_circs_dists = await CreateEnsemblePass.unfold_circ(
                (default_config, 0),
                circuit=circuit,
                pts=pts,
                locations=locations,
                solve_exact_dists=self.solve_exact_dists,
                cost=self.cost,
                target=target
            )
            return [[all_circs_dists]]

        greediest_inds = await self.knapsack_solve(
            psols_diffs, 
            dists, 
            self.success_threshold * 5, 
            self.num_circs * 5,
            greediest=True)
    

        greedier_inds = await self.knapsack_solve(
            psols_diffs, 
            dists, 
            self.success_threshold * 5, 
            self.num_circs * 3,
            greediest=False)

        greedy_inds = await self.knapsack_solve(
            psols_diffs, 
            dists, 
            self.success_threshold * 2, 
            self.num_circs * 3,
            greediest=False)

        safe_greediest_inds = await self.knapsack_solve(
            psols_diffs, 
            dists, 
            self.success_threshold, 
            self.num_circs * 5,
            greediest=True)

        safe_inds = await self.knapsack_solve(
            psols_diffs, 
            dists, 
            self.success_threshold, 
            self.num_circs,
            greediest=False)

        valid_inds = await self.knapsack_solve(
            psols_diffs, 
            dists, 
            self.success_threshold, 
            self.num_circs * 10,
            greediest=False)


        all_inds = [greediest_inds, greedier_inds, greedy_inds, safe_greediest_inds, safe_inds]

        for _ in range(self.num_random_ensembles):
            all_inds.append(random.sample(valid_inds, k=(self.num_circs)))


        print("Got all ensembles", flush=True)

        all_ensembles = []

        for inds in all_inds:
            uppers = []
            all_combos = []

            # Randomly sample a bunch of psols
            for random_inds, total_dist in inds:
                circ_list = [psols[i][ind] for i, ind in enumerate(random_inds)]
                new_config = [CircuitGate(circ) for circ in circ_list]
                all_combos.append((new_config, total_dist))
                uppers.append(total_dist)

            # Add original circuit
            circ_list = [psols[i][-1] for i in range(len(psols))]
            default_config = [CircuitGate(circ) for circ in circ_list]
            all_combos.append((default_config, 0))
            all_circs_dists = await get_runtime().map(
                CreateEnsemblePass.unfold_circ,
                all_combos,
                circuit=circuit,
                pts=pts,
                locations=locations,
                solve_exact_dists=self.solve_exact_dists,
                cost=self.cost,
                target=target
            )


            # init_dists = [dist for circ, dist in all_circs_dists]
            # counts = [circ.count(CNOTGate()) for circ, dist in all_circs_dists]

            # print("Initial Dists: ", init_dists, flush=True)
            # uppers = [dist for circ, dist in all_circs_dists]
            # print("Initial upperbounds: ", uppers, flush=True)
            # print("Initial Counts: ", counts, flush=True)

            all_circs_dists: list[tuple[Circuit, float]] = [(circ, dist) for circ, dist in all_circs_dists if dist < self.success_threshold][:self.num_circs]

            all_circs_dists = sorted(all_circs_dists, key=lambda x: x[0].count(CNOTGate()))

            # print("Selected Dists: ", [dist for circ, dist in all_circs_dists], flush=True)
            if self.sort_by_t:
                final_counts = [circ.num_params for circ, dist in all_circs_dists]
            else:
                final_counts = [circ.count(CNOTGate()) for circ, dist in all_circs_dists]
            # print("Selected Counts: ", [circ.count(CNOTGate()) for circ, dist in all_circs_dists], flush=True)

            if len(all_circs_dists) == 0:
                print(all_circs_dists, flush=True)
                print("No Circuits Found", flush=True)
                all_ensembles.append(None)

            assert len(all_circs_dists) > 0

            if len(all_circs_dists) > 1:
                # Get rid of the default case
                all_circs_dists.pop()

            print("Length of Ensemble", len(all_circs_dists), "Final Average Count: ", np.mean(final_counts), flush=True)
            all_ensembles.append((all_circs_dists, np.mean(final_counts)))

        # Sort by average CNOT count
        all_ensembles = sorted(all_ensembles, key=lambda x: x[1])
        print("Avg Counts", [x[1] for x in all_ensembles], flush=True)
        return [x[0] for x in all_ensembles]


    def parse_data(self,
        blocked_circuit: Circuit,
        data: dict[Any, Any],
    ) -> tuple[list[list[tuple[Circuit, float]]], list[CircuitPoint], list[list[float]]]:
        """Parse the data outputed from synthesis."""
        block_data = data[0]

        psols: list[list[Circuit]] = [[] for _ in block_data]
        dists: list[list[float]] = [[] for _ in block_data]
        pts: list[CircuitPoint] = []
        targets: list[UnitaryMatrix] = []
        thresholds: list[float] = []

        num_sols = 1
        # print("PARSING DATA", flush=True)

        for i, block in enumerate(block_data):
            pts.append(block['point'])
            targets.append(block["target"])
            exact_block: Circuit = blocked_circuit[pts[-1]].gate._circuit.copy()  # type: ignore  # noqa
            exact_block.set_params(blocked_circuit[pts[-1]].params)
            thresholds.append(block["error_percentage_allocated"] * self.success_threshold)

            if 'scan_sols' not in block:
                print("NO SCAN SOLS")
                continue

            psols[i] = [block['scan_sols'][j][0] for j in range(len(block['scan_sols']))]
            dists[i] = [block['scan_sols'][j][1] for j in range(len(block['scan_sols']))]   
            num_sols *= len(psols[i])


        # print([len(psols[i]) for i in range(len(psols))])
        # print("Total Potential Solutions", num_sols)

        self.num_circs = min(self.num_circs, num_sols)

        return psols, pts, dists, targets, thresholds

    async def run(self, circuit: Circuit, data: PassData) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""
        print("Running Ensemble Pass", flush=True)

        # print("Keys", list(data.keys()), flush=True)

        # if "finished_create_ensemble" in data:
        #     print("Ensemble Already Found", flush=True)
        #     return

        # Get scan_sols for each circuit_gate
        block_data = data[ForEachBlockPass.key]


        if self.use_calculated_error:
            self.success_threshold = self.success_threshold * data["error_percentage_allocated"]

        approx_circs, pts, dists, targets, thresholds = self.parse_data(circuit, block_data)
        
        all_ensembles: list[list[tuple[Circuit, float]]] = await self.assemble_circuits(circuit, approx_circs, pts, dists=dists, targets=targets, thresholds=thresholds, target=data.target)

        data["scan_sols"] = []
        data["ensemble"] = []

        for all_circs in all_ensembles:
            all_circs = sorted(all_circs, key=lambda x: x[0].count(CNOTGate()))
            if len(data["scan_sols"]) == 0 and all_circs is not None:
                data["scan_sols"].extend(all_circs)
            if all_circs is not None:
                data["ensemble"].append(all_circs)

        if len(all_ensembles) == 0:
            _logger.error("No ensembles found!!!!")
            return

        if "checkpoint_dir" in data:
            checkpoint_data_file = data["checkpoint_data_file"]
            data["finished_create_ensemble"] = True
            # No longer need block data
            data.pop(ForEachBlockPass.key, None)
            # print("Saving keys", list(data.keys()), flush=True)
            pickle.dump(data, open(checkpoint_data_file, "wb"))
        
        return

        
