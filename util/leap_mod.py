"""This module implements the LEAPSynthesisPass."""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy.stats import linregress
from os.path import join, exists
import pickle
import time

from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import CNOTGate, VariableUnitaryGate, U3Gate
from bqskit.ir.opt.cost.functions import HilbertSchmidtResidualsGenerator
from bqskit.ir.opt.cost.generator import CostFunctionGenerator
from bqskit.passes.search.frontier import Frontier
from bqskit.passes.search.generator import LayerGenerator
from bqskit.passes.search.generators.seed import SeedLayerGenerator
from bqskit.passes.search.generators.simple import SimpleLayerGenerator
from bqskit.passes.search.heuristic import HeuristicFunction
from bqskit.passes.search.heuristics import DijkstraHeuristic
from bqskit.compiler.basepass import BasePass
from bqskit.qis.state.state import StateVector
from bqskit.qis.state.system import StateSystem
from bqskit.qis.unitary import UnitaryMatrix
from bqskit.runtime import get_runtime
from bqskit.utils.typing import is_integer
from bqskit.utils.typing import is_real_number

_logger = logging.getLogger(__name__)


class LEAPSynthesisPass2(BasePass):
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
        cost: CostFunctionGenerator = HilbertSchmidtResidualsGenerator(),
        max_layer: int | None = 40,
        store_partial_solutions: bool = False,
        partials_per_depth: int = 25,
        min_prefix_size: int = 3,
        instantiate_options: dict[str, Any] = {},
        partial_success_threshold: float = 1e-3,
        use_calculated_error: bool = False,
        max_psols: int = 10,
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

            store_partial_solutions (bool): Whether to store partial solutions
                at different depths inside of the data dict. (Default: False)

            partials_per_depth (int): The maximum number of partials
                to store per search depth. No effect if
                `store_partial_solutions` is False. (Default: 25)

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
        self.store_partial_solutions = store_partial_solutions
        self.partials_per_depth = partials_per_depth
        self.max_psols = max_psols


    async def synthesize_circ(
        self,
        utry: UnitaryMatrix | StateVector | StateSystem,
        data: PassData,
        default_circuit: Circuit | None = None,
        frontier: Frontier | None = None,
        save_data_file: str | None = None
    ) -> Circuit:
        """Synthesize `utry`, see :class:`SynthesisPass` for more."""
        # Initialize run-dependent options
        instantiate_options = self.instantiate_options.copy()

        if self.use_calculated_error:
            # use sqrt
            factor = data["error_percentage_allocated"]
            # factor = data
            success_threshold = self.success_threshold * factor
            partial_success_threshold = self.partial_success_threshold * factor
            # print("New Success Threshold", success_threshold, "New Partial Success Threshold", partial_success_threshold)
        else:
            success_threshold = self.success_threshold
            partial_success_threshold = self.partial_success_threshold
       
       # Seed the PRNG
        instantiate_options['ftol'] = partial_success_threshold
        if 'seed' not in instantiate_options:
            instantiate_options['seed'] = data.seed

        block_id = f"Block {data.get('super_block_num', -1)}_{data.get('block_num', -1)}:"
        print(f"{block_id} Partial Success Threshold: ", partial_success_threshold, flush=True)
        print(f"{block_id} Gate Counts: ", default_circuit.gate_counts, flush=True)
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
            best_circ = initial_layer
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
                scan_sols.append((initial_layer.copy(), best_dist))
                scan_sols.append((default_circuit.copy(), 0))
                data['scan_sols'] = scan_sols
                if save_data_file is not None:
                    # Dump data and circuit with empty Frontier
                    data["leap_finished"] = True
                    pickle.dump(data, open(save_data_file, "wb"))
                return initial_layer
            
        else:
            best_dist = data['best_dist']
            best_circ = data['best_circ']
            best_layer = data['best_layer']
            best_dists = data['best_dists']
            best_layers = data['best_layers']
            last_prefix_layer = data['last_prefix_layer']
            scan_sols = data['scan_sols']

        default_count = default_circuit.count(CNOTGate())
        max_layer = min(self.max_layer, default_count + 2)

        # Main loop
        step = 0

        while not frontier.empty():
            # print("CHECKPOINTING!", best_layer, data["block_num"])
            # Save all data
            data['frontier'] = frontier
            data["best_dist"] = best_dist
            data["best_dists"] = best_dists
            data["best_circ"] = best_circ
            data["best_layer"] = best_layer
            data["best_layers"] = best_layers
            data["last_prefix_layer"] = last_prefix_layer
            data["scan_sols"] = scan_sols
            step += 1
            if save_data_file is not None and step % 20 == 0:
                # Dump data and circuit
                pickle.dump(data, open(save_data_file, "wb"))
            
            top_circuit, layer = frontier.pop()

            # if block_num == 0:
            #     print(f"Block {block_id}: Layer {layer} with gate count: {top_circuit.count(CNOTGate())}", flush=True)

            # Generate successors
            successors = layer_gen.gen_successors(top_circuit, data)

            if len(successors) == 0:
                continue
        
            # Instantiate successors
            circuits: list[Circuit] = await get_runtime().map(
                Circuit.instantiate,
                successors,
                target=utry,
                **instantiate_options,
            )

            # Evaluate successors
            for circuit in circuits:
                dist = self.cost.calc_cost(circuit, utry)
                if dist < partial_success_threshold:
                    # print(f"{block_id} Partial Success at layer {layer + 1} with cost: {dist:.12e}", flush=True)
                    scan_sols.append((circuit.copy(), dist))
                    data['scan_sols'] = scan_sols
                    if len(scan_sols) >= self.max_psols:
                        scan_sols.append((default_circuit.copy(), 0))
                        data['scan_sols'] = scan_sols
                        # Save data and circuit
                        if save_data_file is not None:
                            data["leap_finished"] = True
                            pickle.dump(data, open(save_data_file, "wb"))
                        return

                if self.check_new_best(layer + 1, dist, best_layer, best_dist, partial_success_threshold):
                    plural = '' if layer == 0 else 's'
                    _logger.debug(
                        f'{block_id} New best circuit found with {layer + 1} layer{plural}'
                        f' and cost: {dist:.12e}.', flush=True
                    )
                    best_dist = dist
                    best_circ = circuit
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
                        if layer + 1 < max_layer:
                            frontier.add(circuit, layer + 1)

                if layer + 1 < max_layer:
                    frontier.add(circuit, layer + 1)

        if self.store_partial_solutions:
            scan_sols.append((default_circuit.copy(), 0))
            data['scan_sols'] = scan_sols

        # Save data and circuit
        if save_data_file is not None:
            data["leap_finished"] = True
            pickle.dump(data, open(save_data_file, "wb"))

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
        layer_gen = SimpleLayerGenerator(CNOTGate(), single_qudit_gate_1=U3Gate())
        # layer_gen = self.layer_gen or data.gate_set.build_mq_layer_generator()

        # Priority given to seeded synthesis
        if 'seed_circuits' in data:
            return SeedLayerGenerator(data['seed_circuits'], layer_gen)

        return layer_gen

    async def run(self, circuit: Circuit, data: PassData) -> None:
        """Perform the pass's operation, see :class:`BasePass` for more."""
        print(f"Starting LEAP for block {data.get('super_block_num', -1)} : {data.get('block_num', -1)}", flush=True)
        if "leap_finished" in data and data["leap_finished"]:
            print("LEAP is already finished!", flush=True)
            return

        frontier = None
        save_data_file = None
        save_data_file = data.get("checkpoint_data_file", None)
        if save_data_file:
            assert(exists(save_data_file))
            frontier: Frontier | None = data.get('frontier', None)
            leap_finished = data.get('leap_finished', False)
            _logger.debug(f'Loading data from {save_data_file}')
            if frontier is not None:
                _logger.debug(f'Frontier is empty: {frontier.empty()}')
            if leap_finished:
                _logger.debug('Block is already finished!')
                return
            data['leap_finished'] = False
    
        await self.synthesize_circ(data.target, data, default_circuit=circuit, frontier=frontier, save_data_file=save_data_file)