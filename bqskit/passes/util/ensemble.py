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
from bqskit.ir.opt.cost.functions import FrobeniusCostGenerator
from bqskit.ir.opt.minimizers.lbfgs import LBFGSMinimizer
from bqskit.ir.opt.cost.generator import CostFunctionGenerator
from bqskit.ir.gates import CircuitGate
from bqskit.ir.gates import CNOTGate, FixedRZGate
from bqskit.qis import UnitaryMatrix
import numpy as np
import pickle
import os


_logger = logging.getLogger(__name__)

frob_cost = FrobeniusCostGenerator()

class CreateEnsemblePass(BasePass):
    """Converts single-qubit general unitary gates to U3 Gates."""

    finished_pass_str = "finished_create_ensemble"

    def __init__(self, success_threshold = 1e-4, 
                 num_circs = 1000,
                 cost: CostFunctionGenerator = FrobeniusCostGenerator(),
                 use_calculated_error: bool = False,
                 num_random_ensembles: int = 3,
                 solve_exact_dists: bool = False,
                 sort_by_t: bool = False,
                 checkpoint_extra_str: str = "") -> None:
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
        self.checkpoint_extra_str = checkpoint_extra_str

    async def unfold_circ(
            config_dist: tuple[list[CircuitGate], float], 
            circuit: Circuit, 
            pts: list[CircuitPoint], 
            locations: list[CircuitLocationLike], 
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

        dist = frob_cost.calc_cost(copied_circuit, target)

        return copied_circuit, dist

    def get_random_inds(self, dists: list[list[float]], num_circs: int) -> list[list[int]]:
        '''
        Get all possible solutions for the ensemble up to 
        '''
        num_psols_per_block = [len(psol_dists) for psol_dists in dists]

        # Get all indice combos from all blocks, picking one psol from each block
        all_inds = np.zeros((num_circs, len(num_psols_per_block)), dtype=int)
        for i, num_psols in enumerate(num_psols_per_block):
            # Get num_circs random inds
            all_inds[:, i] = np.random.choice(num_psols, 
                                              size=num_circs, 
                                              replace=True)
        return all_inds

    async def knapsack_solve(self, psol_diffs: list[list[int]], psol_dists: list[list[float]], total_dist: float, num_circs: int) -> list:
        '''
        Pick an ensemble of circuits that minimizes the total number of CNOTs while keeping the total distance below a threshold
        You must pick one psol from each list of psol_diffs.
        

        Args:
            psol_diffs: list of lists of differences in CNOT/T counts 
            dists: list of lists of distances
            total_dist: total distance allowed

        Returns:
            list of list of indices to pick. Each list corresponds to a list of psols. 
            There are `num_circs` lists in total.
        
        '''

        # Most greedy algorithm
        # Pick the psol that minimizes the difference in CNOT counts
        # Diffs: Matrix of CNOT count diffs
        # Dists: Matrix of distances

        # We are just padding the arrays that were given with 
        gate_count_diffs = np.zeros(shape=[len(psol_diffs),len(max(psol_diffs,key = lambda x: len(x)))])
        dists = np.zeros_like(gate_count_diffs)

        for i,j in enumerate(psol_diffs):
            gate_count_diffs[i][0:len(j)] = j

        for i,j in enumerate(psol_dists):
            dists[i][0:len(j)] = j  

        # Pick a random ordering to search the psols
        # Weighted by max CNOT difference
        weight = np.array([np.min(x) for x in psol_diffs]) + ((np.min(gate_count_diffs) - 0.1) / 10)
        # print("Weight", weight, flush=True)
        weight = weight / np.sum(weight)
        orig_inds = np.random.choice(len(gate_count_diffs), 
                                     size=(len(gate_count_diffs)),
                                     p=weight,
                                     replace=False)
        
        # print("Inds we are looking at: ", orig_inds)
        # print("Number of psols per block: ", [len(psol_diffs[i]) for i in orig_inds], flush=True)
        # print("Avg CNOT diffs per block: ", [np.mean(psol_diffs[i]) for i in orig_inds], flush=True)
        # print("Avg Dists per block: ", [np.mean(psol_dists[i]) for i in orig_inds], flush=True)
        # print("Avg Total Dist: ", sum([np.mean(psol_dists[i]) for i in orig_inds]), flush=True)
        # print("Max Total Dist: ", sum([np.max(psol_dists[i]) for i in orig_inds]), flush=True)
        # print("Distance Threshold: ", total_dist, flush=True)

        # Get all possible solutions along with their CNOT count differences
        all_valid_inds = self.BFS(orig_inds, psol_diffs, dists, total_dist)

        # print("INDS: ", flush=True)
        # print(["-".join([str(y) for y in x[0]]) for x in all_valid_inds], flush=True)

        # print("NUM VALID INDS", len(all_valid_inds), flush=True)

        # Sort the solutions by the number of CNOTs saved
        sorted_inds = sorted(all_valid_inds, key=lambda x: x[1])
        weights = np.array([x[1] for x in sorted_inds])
        # Remove all positive weights
        weights = weights - np.max(weights)
        # Remove all zeros
        weights = weights - 0.0001
        weights = weights / np.sum(weights)
        # Add uniform distribution to encourage exploration
        weights += 3 / len(weights)
        weights = weights / np.sum(weights)
        try:
            random_inds = np.random.choice(len(sorted_inds), size=min(num_circs, len(sorted_inds)), p=weights, replace=False)
        except Exception as e:
            print("Weights", weights, flush=True)
            print("Sorted Inds", sorted_inds, flush=True)
            _logger.error(e)
        
        # Return a list with 2 things:
        # 1. List of indices to pick (psol index for each block)
        # 2. Total distance
        return [(sorted_inds[i][0]) for i in random_inds]


    def BFS(self, orig_inds, gate_count_diffs, dists, total_dist) -> list[tuple[list[int], int, float]]:
        '''
        Given a total distance, do a BFS to find a subset of
        psols that meets the distance constraint and tries to minimize
        the number of CNOTs.
        
        '''
        # The queue holds the blocks we still have to search over,
        # The current distance,
        # The psol ind for each block we have chosen so far,
        # The current mean_unitary,
        # and the current number of CNOTs/T gates saved
        queue = deque([])
        queue.append((orig_inds, 0, [-1 for _ in gate_count_diffs], 0))
        final_inds = []
        max_total_sols = 1000
        while len(queue) > 0:
            inds, cur_dist, psol_inds, cnots_saved = queue.popleft()
            if len(inds) == 0:
                # If there are no more inds to search over,
                # Add the currents set of psols as a valid solution
                final_inds.append((psol_inds, cnots_saved, cur_dist))
            else:
                # Get the next psol we are considering
                ind = inds[0]
                pos_sols = 0
                
                # Otherwise rank the psols randomly, weighted by CNOT diff
                weight = np.array(gate_count_diffs[ind])
                # Adjust to get rid of positive weights
                weight = weight - np.max(weight)
                # Get rid of 0s
                weight = weight + ((np.min(gate_count_diffs[ind]) - 0.1) / 10)
                weight = weight / np.sum(weight)
                # print(f"Weight[{ind}]", weight, flush=True)
                cols = np.random.choice(len(gate_count_diffs[ind]), 
                                        size=(len(gate_count_diffs[ind])),
                                        p = weight, 
                                        replace=False)
                # Find which psols do not go over the total distance constraint
                # Make sure that it decreases
                new_dists = []
                for j in cols:
                    dist = dists[ind][j]
                    new_dist = cur_dist + dist
                    if new_dist <= total_dist:
                        pos_sols += 1
                    new_dists.append(new_dist)

                # If we have too many solutions, only explore one
                if len(queue) > max_total_sols:
                    # Only explore one solution
                    pos_sols = 1
                else:
                    # Otherwise, explore up to 5 solutions
                    pos_sols = min(pos_sols, 5)

                for j in cols:
                    dist = dists[ind][j]
                    new_dist = cur_dist + dist
                    if new_dist <= total_dist:
                        # print("New Dist: ", new_dist, flush=True)
                        pos_sols -= 1
                        new_psol_inds = psol_inds.copy()
                        new_psol_inds[ind] = int(j)
                        new_cnots_saved = cnots_saved + gate_count_diffs[ind][j]
                        queue.append((inds[1:], new_dist, new_psol_inds, new_cnots_saved))
                    if pos_sols == 0:
                        break

        return final_inds

    async def assemble_circuits(
        self,
        circuit: Circuit,
        psols: list[list[Circuit]],
        pts: list[CircuitPoint],
        dists: list[list[float]],
        target: UnitaryMatrix = None
    ) -> list[list[tuple[Circuit, float]]]:
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
                target=target
            )
            return [[all_circs_dists]]

        # Try to fill a knapsack with a combo of the block
        # psols that minimizes the number of CNOTs while
        # keeping the total distance below a threshold

        # We can target a few different threshold distances
        # Since the upperbound by adding distances is 
        # pretty loose

        greediest_inds = await self.knapsack_solve(
            psols_diffs, 
            dists, 
            self.success_threshold * 5, 
            self.num_circs * 3)

        greedy_inds = await self.knapsack_solve(
            psols_diffs, 
            dists, 
            self.success_threshold * 1.5, 
            self.num_circs * 2)

        valid_inds = await self.knapsack_solve(
            psols_diffs, 
            dists, 
            self.success_threshold, 
            self.num_circs * 2)
        
        # Randomly select a group of solutions
        print("Getting ALl Inds", flush=True)
        # all_inds = self.get_all_inds(dists)
        # selection = np.random.choice(len(all_inds), size=min(self.num_circs, possible_sols, len(all_inds)), replace=False)
        # random_inds = np.array(all_inds)[selection]
        all_inds = [greediest_inds, greedy_inds, valid_inds]
        for _ in range(self.num_random_ensembles):
            random_inds = self.get_random_inds(dists, num_circs=self.num_circs*2)
            all_inds.append(random_inds)


        all_ensembles = []

        for i, inds in enumerate(all_inds):
            ### Get all corresponding circuits
            all_combos = []

            for psol_inds in inds:
                circ_list = [psols[i][ind] for i, ind in enumerate(psol_inds)]
                new_config = [CircuitGate(circ) for circ in circ_list]
                all_combos.append((new_config, self.success_threshold))
            
            # Add default config so at least one solution works
            circ_list = [psols[i][-1] for i in range(len(psols))]
            default_config = [CircuitGate(circ) for circ in circ_list]
            all_combos.append((default_config, 0))

            all_circs_dists = await get_runtime().map(
                CreateEnsemblePass.unfold_circ,
                all_combos,
                circuit=circuit,
                pts=pts,
                locations=locations,
                target=target
            )

            #### Get Valid Circuits with distance < threshold
            valid_circs_dists: list[tuple[Circuit, float]] = [(circ, dist) for circ, dist in all_circs_dists if dist < self.success_threshold]
            valid_circs_dists = sorted(valid_circs_dists, key=lambda x: x[0].count(CNOTGate()))

            print("Number of Valid Circuits", len(valid_circs_dists), flush=True)

            # Trim down to self.num_circs randomly
            if len(valid_circs_dists) > self.num_circs:
                final_random_inds = np.random.choice(len(valid_circs_dists), 
                                                     size=self.num_circs, 
                                                     replace=False)
                
                valid_circs_dists = [valid_circs_dists[i] for i in final_random_inds]

            if self.sort_by_t:
                final_counts = [circ.num_params for circ, _ in valid_circs_dists]
            else:
                final_counts = [circ.count(CNOTGate()) for circ, _ in valid_circs_dists]

            all_ensembles.append((valid_circs_dists, np.mean(final_counts)))

        # Sort by average gate count
        print("Ensemble Sizes", [len(x[0]) for x in all_ensembles], flush=True)
        print("Ensemble Counts", [x[1] for x in all_ensembles], flush=True)
        all_ensembles = sorted(all_ensembles, key=lambda x: x[1])
        return [x[0] for x in all_ensembles]

    def parse_data(self,
        blocked_circuit: Circuit,
        data: dict[Any, Any],
    ) -> tuple[list[list[tuple[Circuit, float]]], list[CircuitPoint], list[list[float]]]:
        """Parse the data outputed from synthesis."""
        block_data = data[0]


        # Look up to get data for each block psol
        # index 1 is the block index (which sub-block in current large block)
        # index 2 is the index of the psol in the block
        psols: list[list[Circuit]] = [[] for _ in block_data]
        dists: list[list[float]] = [[] for _ in block_data]
        # self.psol_unitaries: list[list[UnitaryMatrix]] = []

        # Data to reconstruct the circuit later 
        pts: list[CircuitPoint] = []

        # Deubgging ingo
        targets: list[UnitaryMatrix] = []
        thresholds: list[float] = [[] for _ in block_data]

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
            # self.psol_unitaries.append([psol.get_unitary() for psol in psols[i]])
            num_sols *= len(psols[i])


        # print([len(psols[i]) for i in range(len(psols))])
        print("Total Potential Solutions", num_sols)

        self.num_circs = min(self.num_circs, num_sols)

        return psols, pts, dists, targets, thresholds

    async def run(self, circuit: Circuit, data: PassData) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""
        print("Running Ensemble Pass on block", data.get("block_num", -1), flush=True)

        checkpoint_str = CreateEnsemblePass.finished_pass_str + self.checkpoint_extra_str

        if checkpoint_str in data:
            print("Finished Create Ensemble", flush=True)
            return

        # Get scan_sols for each circuit_gate
        block_data = data[ForEachBlockPass.key]

        if self.use_calculated_error:
            self.success_threshold = self.success_threshold * data["error_percentage_allocated"]


        # See if ensemble pickle exists
        # save_data_file = data["checkpoint_data_file"]
        # ensemble_file = save_data_file.replace(".data", "_scan_ensemble.pkl")
        # print("ENSEMBLE FILE", ensemble_file, flush=True)
        # if os.path.exists(ensemble_file):
        #     print("Using SCANNING GATE ENSEMBLE!", flush=True)
        #     scan_sols = pickle.load(open(ensemble_file, "rb"))
        #     all_circ_dists = [(circ, frob_cost.calc_cost(circ, data.target)) for circ, _ in scan_sols]
        #     all_circ_dists = sorted(all_circ_dists, key=lambda x: x[0].count(CNOTGate()))
        #     data["ensemble"] = [all_circ_dists]
        #     data["scan_sols"] = [all_circ_dists]
        # else:
        data["scan_sols"] = []
        data["ensemble"] = []
            
        approx_circs, pts, dists, targets, thresholds = self.parse_data(circuit, block_data)        
        all_ensembles = await self.assemble_circuits(circuit, approx_circs, pts, dists=dists, target=data.target)

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
            data[checkpoint_str] = True
            # No longer need block data
            # data.pop(ForEachBlockPass.key, None)
            # print("Saving keys", list(data.keys()), flush=True)
            pickle.dump(data, open(checkpoint_data_file, "wb"))
        
        return

        
