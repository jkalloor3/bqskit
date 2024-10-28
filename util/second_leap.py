"""This module implements the LEAPSynthesisPass."""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy.stats import linregress
from os.path import join, exists
import pickle
from pathlib import Path

from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import CNOTGate
from bqskit.ir.opt.cost.functions import HilbertSchmidtResidualsGenerator, HSCostGenerator
from bqskit.ir.opt.cost.generator import CostFunctionGenerator
from bqskit.passes.search.frontier import Frontier
from bqskit.passes.control import ForEachBlockPass
from bqskit.passes.search.generator import LayerGenerator
from bqskit.passes.search.generators.seed import SeedLayerGenerator
from bqskit.passes.search.heuristic import HeuristicFunction
from bqskit.passes.search.heuristics import AStarHeuristic, DijkstraHeuristic
from bqskit.compiler.basepass import BasePass
from bqskit.qis.state.state import StateVector
from bqskit.qis.state.system import StateSystem
from bqskit.qis.unitary import UnitaryMatrix
from bqskit.runtime import get_runtime
from bqskit.utils.typing import is_integer
from bqskit.utils.typing import is_real_number
from itertools import chain, zip_longest

_logger = logging.getLogger(__name__)


class SecondLEAPSynthesisPass(BasePass):
    """
    A pass implementing the LEAP search synthesis algorithm.

    References:
        Ethan Smith, Marc G. Davis, Jeffrey M. Larson, Ed Younis,
        Lindsay Bassman, Wim Lavrijsen, and Costin Iancu. 2022. LEAP:
        Scaling Numerical Optimization Based Synthesis Using an
        Incremental Approach. ACM Transactions on Quantum Computing
        (June 2022). https://doi.org/10.1145/3548693
    """

    def __init__(
        self,
        heuristic_function: HeuristicFunction = DijkstraHeuristic(),
        layer_generator: LayerGenerator | None = None,
        success_threshold: float = 1e-8,
        cost: CostFunctionGenerator = HSCostGenerator(),
        max_layer: int | None = None,
        min_prefix_size: int = 3,
        instantiate_options: dict[str, Any] = {},
        partial_success_threshold: float = 1e-3,
        use_calculated_error: bool = False,
        max_psols: int = 5,
    ) -> None:
        """
        Construct a search-based synthesis pass.

        Args:
            heuristic_function (HeuristicFunction): The heuristic to guide
                search.

            layer_generator (LayerGenerator | None): The successor function
                to guide node expansion. If left as none, then a default
                will be selected before synthesis based on the target
                model's gate set. (Default: None)

            success_threshold (float): The distance threshold that
                determines successful termintation. Measured in cost
                described by the cost function. (Default: 1e-8)

            cost (CostFunction | None): The cost function that determines
                distance during synthesis. The goal of this synthesis pass
                is to implement circuits for the given unitaries that have
                a cost less than the `success_threshold`.
                (Default: HSDistance())

            max_layer (int): The maximum number of layers to append without
                success before termination. If left as None it will default
                to unlimited. (Default: None)

            min_prefix_size (int): The minimum number of layers needed
                to prefix the circuit.

            instantiate_options (dict[str: Any]): Options passed directly
                to circuit.instantiate when instantiating circuit
                templates. (Default: {})

        Raises:
            ValueError: If `max_depth` or `min_prefix_size` is nonpositive.
        """
        if not isinstance(heuristic_function, HeuristicFunction):
            raise TypeError(
                'Expected HeursiticFunction, got %s.'
                % type(heuristic_function),
            )

        if layer_generator is not None:
            if not isinstance(layer_generator, LayerGenerator):
                raise TypeError(
                    f'Expected LayerGenerator, got {type(layer_generator)}.',
                )

        if not is_real_number(success_threshold):
            raise TypeError(
                'Expected real number for success_threshold'
                ', got %s' % type(success_threshold),
            )

        if not isinstance(cost, CostFunctionGenerator):
            raise TypeError(
                'Expected cost to be a CostFunctionGenerator, got %s'
                % type(cost),
            )

        if max_layer is not None and not is_integer(max_layer):
            raise TypeError(
                'Expected max_layer to be an integer, got %s' % type(max_layer),
            )

        if max_layer is not None and max_layer <= 0:
            raise ValueError(
                'Expected max_layer to be positive, got %d.' % int(max_layer),
            )

        if min_prefix_size is not None and not is_integer(min_prefix_size):
            raise TypeError(
                'Expected min_prefix_size to be an integer, got %s'
                % type(min_prefix_size),
            )

        if min_prefix_size is not None and min_prefix_size <= 0:
            raise ValueError(
                'Expected min_prefix_size to be positive, got %d.'
                % int(min_prefix_size),
            )

        if not isinstance(instantiate_options, dict):
            raise TypeError(
                'Expected dictionary for instantiate_options, got %s.'
                % type(instantiate_options),
            )

        self.heuristic_function = heuristic_function
        self.layer_gen = layer_generator
        self.success_threshold = success_threshold
        self.partial_success_threshold = partial_success_threshold
        self.cost = cost
        self.max_layer = max_layer
        self.min_prefix_size = min_prefix_size
        self.instantiate_options: dict[str, Any] = {
            'cost_fn_gen': HilbertSchmidtResidualsGenerator(),
        }
        self.use_calculated_error = use_calculated_error
        self.instantiate_options.update(instantiate_options)
        self.max_psols = max_psols

    async def synthesize(self, data: PassData, target: UnitaryMatrix, default_circuit: Circuit, datas: list[PassData], save_data_files: list[str] = None) -> Circuit:
        # Synthesize every circuit in the ensemble
        circs: list[Circuit] = [d[0] for d in data['scan_sols'][:-1]]
        if len(circs) > 0:
            for c in circs:
                assert(isinstance(c, Circuit))

            utries = [(c.get_unitary(), datas[i], save_data_files[i]) for i, c in enumerate(circs)]

            new_circs: list[list[Circuit]] = await get_runtime().map(
                self.synthesize_circ,
                utries,
            )
            for i, circs in enumerate(new_circs):
                if not isinstance(circs, list):
                    print(len(utries), len(new_circs), len(circs), type(circs), flush=True)
                assert(isinstance(circs, list))
                for c in circs:
                    assert(isinstance(c, Circuit))

            # max_cnot_gates = default_circuit.count(CNOTGate())

            for i, circs in enumerate(new_circs):
                new_circs[i] = sorted(circs, key=lambda x: x.count(CNOTGate()))
                # for j, c in enumerate(new_circs[i]):
                    # if c.count(CNOTGate()) >= max_cnot_gates: # Only take shorter circuits
                    #     new_circs[i][j] = None
            
            # Now interleave so you get a decent mix
            new_circs = list(chain(*zip_longest(*new_circs, fillvalue=None)))
            new_circs: list[tuple[Circuit, float]] = [(c, self.cost(c, target)) for c in new_circs if c is not None]
            new_circs = sorted(new_circs, key=lambda x: x[0].count(CNOTGate()))
        else:
            new_circs = []

        # Randomly choose up to 15 circuits
        if len(new_circs) > 15:
            inds = np.random.choice(len(new_circs), 15, replace=False)
            print("Random inds", inds, flush=True)
            new_circs = [new_circs[i] for i in inds]

        new_circs.append((default_circuit, 0))

        assert(isinstance(new_circs[0][0], Circuit))

        assert(len(new_circs) > 0)

        # print("Return Scan Sols")
        data['scan_sols'] = new_circs

        # Save circuit
        # if self.save_circuit_file:
        #     pickle.dump(new_circs[0][0], open(self.save_circuit_file, "wb"))

        return new_circs[0][0]

    async def synthesize_circ(
        self,
        utry_data: tuple[UnitaryMatrix | StateVector | StateSystem, PassData, str | None],
    ) -> list[Circuit]:
        """Synthesize `utry`, see :class:`SynthesisPass` for more."""

        utry, data, save_data_file = utry_data
        if self.use_calculated_error:
            # use sqrt
            factor = np.sqrt(data["error_percentage_allocated"]) * 2 
            # factor = data["error_percentage_allocated"]
            print("Percentage Allocated: ", factor, flush=True)
            success_threshold = self.success_threshold * factor
            partial_success_threshold = self.partial_success_threshold * factor
        else:
            success_threshold = self.success_threshold
            partial_success_threshold = self.partial_success_threshold

        # Initialize run-dependent options
        instantiate_options = self.instantiate_options.copy()
        instantiate_options['ftol'] = success_threshold

        frontier = data.get("frontier", None)

        if data.get("second_leap_finished", False):
            _logger.debug('Block is already finished!')
            # print(f"Block is finished!", flush=True)
            return data['scan_sols']

        # Seed the PRNG
        if 'seed' not in instantiate_options:
            instantiate_options['seed'] = data.seed

        # Get layer generator for search
        layer_gen = self._get_layer_gen(data)

        if frontier is None:
            # Begin the search with an initial layer
            frontier = Frontier(utry, self.heuristic_function)
            initial_layer = layer_gen.gen_initial_layer(utry, data)
            initial_layer.instantiate(utry, **instantiate_options)

            frontier.add(initial_layer, 0)

            # Track best circuit, initially the initial layer
            best_dist = self.cost.calc_cost(initial_layer, utry)
            best_layer = 0
            best_dists = [best_dist]
            best_layers = [0]
            last_prefix_layer = 0

            # Track partial solutions
            scan_sols: list[tuple[Circuit, float]] = []

            _logger.debug(f'Search started, initial layer has cost: {best_dist}.')

            # Evalute initial layer
            if best_dist < success_threshold:
                _logger.debug('Successful synthesis.')
                scan_sols.append(initial_layer.copy())
                data['scan_sols'] = scan_sols
                if save_data_file:
                    # Dump data and circuit with empty Frontier
                    data["second_leap_finished"] = True
                    pickle.dump(data, open(save_data_file, "wb"))
                return scan_sols
            
        else:
            best_dist = data['best_dist']
            best_layer = data['best_layer']
            best_dists = data['best_dists']
            best_layers = data['best_layers']
            last_prefix_layer = data['last_prefix_layer']
            scan_sols = data['scan_sols']

        _logger.debug(f'Search started, initial layer has cost: {best_dist}.')

        # Evalute initial layer
        # if best_dist < self.success_threshold:
        #     _logger.debug('Successful synthesis.')
        #     data['scan_sols'] = scan_sols
        #     if save_data_file is not None:
        #         # Dump data and circuit with empty Frontier
        #         frontier.pop()
        #         data['frontier'] = frontier
        #         pickle.dump(data, open(save_data_file, "wb"))
        #     return [scan_sols[0]]

        # Main loop
        step = 0
        while not frontier.empty():
            # print("Iteration", step, best_layer, data.get("block_num", 0), flush=True)
            data['frontier'] = frontier
            data["best_dist"] = best_dist
            data["best_dists"] = best_dists
            data["best_layer"] = best_layer
            data["best_layers"] = best_layers
            data["last_prefix_layer"] = last_prefix_layer
            data["scan_sols"] = scan_sols
            step += 1
            if save_data_file and step % 10 == 0:
                # Dump data and circuit
                # print("Checkpointing!")
                pickle.dump(data, open(save_data_file, "wb"))

            top_circuit, layer = frontier.pop()

            # Generate successors
            successors = layer_gen.gen_successors(top_circuit, data)

            if len(successors) == 0:
                continue

            # Instantiate successors
            circuits = await get_runtime().map(
                Circuit.instantiate,
                successors,
                target=utry,
                **instantiate_options,
            )

            # Evaluate successors
            for circuit in circuits:
                dist = self.cost.calc_cost(circuit, utry)

                if dist < partial_success_threshold:
                    scan_sols.append(circuit.copy())
                    data['scan_sols'] = scan_sols
                    if len(scan_sols) >= self.max_psols:
                        # return here
                        # Save data and circuit
                        if save_data_file is not None:
                            data["second_leap_finished"] = True
                            pickle.dump(data, open(save_data_file, "wb"))
                        return scan_sols

                if dist < success_threshold and len(scan_sols) >= self.max_psols:
                    _logger.debug(f'Successful synthesis with distance {dist:.6e}.')
                    if save_data_file:
                        data["second_leap_finished"] = True
                        data["scan_sols"] = scan_sols
                        pickle.dump(data, open(save_data_file, "wb"))
                    return scan_sols

                if self.check_new_best(layer + 1, dist, best_layer, best_dist, success_threshold):
                    plural = '' if layer == 0 else 's'
                    _logger.debug(
                        f'New best circuit found with {layer + 1} layer{plural}'
                        f' and cost: {dist:.12e}.',
                    )
                    best_dist = dist
                    best_layer = layer + 1

                    if self.check_leap_condition(
                        layer + 1,
                        best_dist,
                        best_layers,
                        best_dists,
                        last_prefix_layer,
                    ):
                        _logger.debug(f'Prefix formed at {layer + 1} layers.')
                        last_prefix_layer = layer + 1
                        frontier.clear()
                        if self.max_layer is None or layer + 1 < self.max_layer:
                            frontier.add(circuit, layer + 1)

                if self.max_layer is None or layer + 1 < self.max_layer:
                    frontier.add(circuit, layer + 1)

        _logger.warning('Frontier emptied.')
        _logger.warning(
            'Returning best known circuit with %d layer%s and cost: %e.'
            % (best_layer, '' if best_layer == 1 else 's', best_dist),
        )

        if save_data_file:
            data["second_leap_finished"] = True
            data["scan_sols"] = scan_sols
            pickle.dump(data, open(save_data_file, "wb"))
        return scan_sols

    def check_new_best(
        self,
        layer: int,
        dist: float,
        best_layer: int,
        best_dist: float,
        success_threshold: float,
    ) -> bool:
        """
        Check if the new layer depth and dist are a new best node.

        Args:
            layer (int): The current layer in search.

            dist (float): The current distance in search.

            best_layer (int): The current best layer in the search tree.

            best_dist (float): The current best distance in search.
        """
        better_layer = (
            dist < best_dist
            and (
                best_dist >= success_threshold
                or layer <= best_layer
            )
        )
        better_dist_and_layer = (
            dist < success_threshold and layer < best_layer
        )
        return better_layer or better_dist_and_layer

    def check_leap_condition(
        self,
        new_layer: int,
        best_dist: float,
        best_layers: list[int],
        best_dists: list[float],
        last_prefix_layer: int,
    ) -> bool:
        """
        Return true if the leap condition is satisfied.

        Args:
            new_layer (int): The current layer in search.

            best_dist (float): The current best distance in search.

            best_layers (list[int]): The list of layers associated
                with recorded best distances.

            best_dists (list[float]): The list of recorded best
                distances.

            last_prefix_layer (int): The last layer a prefix was formed.
        """

        with np.errstate(invalid='ignore', divide='ignore'):
            # Calculate predicted best value
            m, y_int, _, _, _ = linregress(best_layers, best_dists)

        predicted_best = m * (new_layer) + y_int

        # Track new values
        best_layers.append(new_layer)
        best_dists.append(best_dist)

        if np.isnan(predicted_best):
            return False

        # Compute difference between actual value
        delta = predicted_best - best_dist

        _logger.debug(
            'Predicted best value %f for new best best with delta %f.'
            % (predicted_best, delta),
        )

        layers_added = new_layer - last_prefix_layer
        return delta < 0 and layers_added >= self.min_prefix_size

    def _get_layer_gen(self, data: PassData) -> LayerGenerator:
        """
        Set the layer generator.

        If a layer generator has been passed into the constructor, then that
        layer generator will be used. Otherwise, a default layer generator will
        be selected by the gateset.

        If seeds are passed into the data dict, then a SeedLayerGenerator will
        wrap the previously selected layer generator.
        """
        # TODO: Deduplicate this code with qsearch synthesis
        layer_gen = self.layer_gen or data.gate_set.build_mq_layer_generator()

        # Priority given to seeded synthesis
        if 'seed_circuits' in data:
            return SeedLayerGenerator(data['seed_circuits'], layer_gen)

        return layer_gen

    async def run(self, circuit: Circuit, data: PassData) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""
        datas: list[PassData] = [data.copy() for _ in data['scan_sols']]
        save_file: str = data.get("checkpoint_data_file", None)
        if save_file:
            # File is of the form "checkpoint_dir/block_{num}.data"
            # We want a separate file for each psol in block of form
            # "checkpoint_dir/block_{num}_{i}.data"
            save_data_files = [save_file.replace(".data", f"_{i}.data") for i in range(len(data['scan_sols']))]
            _logger.debug("Second LEAP: Saving data files from %s", save_data_files)
        else:
            save_data_files = [None for _ in data['scan_sols']]

        for i, save_data_file in enumerate(save_data_files):
            if save_data_file and not exists(save_data_file):
                with open(save_data_file, 'wb') as df:
                    datas[i]['frontier'] = None
                    pickle.dump(datas[i], df)
            elif save_data_file and exists(save_data_file):
                with open(save_data_file, 'rb') as df:
                    datas[i] = pickle.load(df)
                frontier: Frontier | None = datas[i].get("frontier", None)
                if frontier is None:
                    _logger.debug("How is this possible for frontier?")
                else:
                    _logger.debug(f"Second leap frontier {i} is empty: {frontier.empty()}")

        circuit.become(await self.synthesize(data, target=data.target, default_circuit=data['scan_sols'][-1][0], datas=datas, save_data_files=save_data_files))

        # Clean up data files after finishing
        # for save_data_file in save_data_files:
        #     if save_data_file and exists(save_data_file):
        #         print(f"Deleting {save_data_file}", flush=True)
        #         Path(save_data_file).unlink()