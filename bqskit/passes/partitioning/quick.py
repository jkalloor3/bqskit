"""This module defines the QuickPartitioner pass."""
from __future__ import annotations

import heapq
import logging
from typing import Any, List

import networkx as nx

from bqskit.compiler.basepass import BasePass
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates.circuitgate import CircuitGate
from bqskit.ir.region import CircuitRegion
from bqskit.utils.typing import is_integer

_logger = logging.getLogger(__name__)


class QuickPartitioner(BasePass):
    """
    The QuickPartitioner Pass.

    This pass forms partitions in the circuit by iterating over the operations
    in a topological order and binning them into blocks.
    """

    def __init__(
        self,
        block_size: int = 3,
    ) -> None:
        """
        Construct a QuickPartitioner.

        Args:
            block_size (int): Maximum size of partitioned blocks.
                (Default: 3)

        Raises:
            ValueError: If `block_size` is less than 2.
        """

        if not is_integer(block_size):
            raise TypeError(
                f'Expected integer for block_size, got {type(block_size)}.',
            )

        if block_size < 2:
            raise ValueError(
                f'Expected block_size to be greater than 2, got {block_size}.',
            )

        self.block_size = block_size
        self.edges = []

    def run(self, circuit: Circuit, data: dict[str, Any] = {}) -> None:
        """
        Partition gates in a circuit into a series of CircuitGates.

        Args:
            circuit (Circuit): Circuit to be partitioned.

            data (dict[str,Any]): Optional data unique to specific run.
        """

        # Number of qudits in the circuit
        num_qudits = circuit.num_qudits

        # If block size > circuit size, return the circuit as a block
        if self.block_size > num_qudits:
            _logger.warning(
                'Configured block size is greater than circuit size; '
                'blocking entire circuit.',
            )
            circuit.fold({
                qudit_index: (0, circuit.num_cycles)
                for qudit_index in range(circuit.num_qudits)
            })
            return

        # List to hold the active blocks
        active_blocks: Any = []

        # List to hold the finished blocks
        finished_blocks: Any = {}
        block_id = 0

        # Active qudit cycles and block-qudit dependencies
        qudit_actives: Any = [{} for _ in range(num_qudits)]
        qudit_dependencies: Any = [{} for _ in range(num_qudits)]

        # The partitioned circuit
        partitioned_circuit = Circuit(num_qudits, circuit.radixes)

        total_new_regions = 0
        total_regions = 0

        # For each cycle, operation in topological order
        for cycle, op in circuit.operations_with_cycles():

            # Get the qudits of the operation
            qudits = op.location._location

            # Update the active locations of the qudits
            for qudit in qudits:
                qudit_actives[qudit][cycle] = None

            # Compile a list of admissible blocks out of the
            # active blocks for the operation
            admissible_blocks = []
            for index, block in enumerate(active_blocks):
                if all([qudit not in block[-1] for qudit in qudits]):
                    admissible_blocks.append(index)

            # Boolean indicator to capture if an active block
            # has been found for the operation
            found = False

            # For all admissible blocks, check if all operation
            # qudits are in the block. If such a block is found,
            # update the upper region bound for the corresponding
            # qudits, and raise the found boolean
            for block in [active_blocks[index] for index in admissible_blocks]:
                if all([qudit in block for qudit in qudits]):
                    for qudit in qudits:
                        block[qudit][1] = cycle
                    found = True
                    break

            updated_qudits = set()

            # If such a block is not found
            if not found:

                # For all admissible blocks, check if the operation
                # qudits can be added to the block without breaching
                # the size limit. If such a block is found, add the
                # new qudits, update the region bounds, check if any
                # blocks are finished, and raise the found boolean
                for block in [
                    active_blocks[index]
                    for index in admissible_blocks
                ]:
                    if len(set(list(qudits) + list(block.keys()))) - \
                            1 <= self.block_size:
                        for qudit in qudits:
                            if qudit not in block:
                                block[qudit] = [cycle, cycle]
                            else:
                                block[qudit][1] = cycle
                        block_id, updated_qudits = self.compute_finished_blocks(
                            block, qudits, active_blocks,
                            finished_blocks, block_id, qudit_dependencies, cycle, num_qudits,  # noqa
                        )
                        found = True
                        break

            # If a block is still not found, check if any blocks are finished
            # with the new operation qudits, create a new block, and add it
            # to the list of active blocks
            if not found:
                block_id, updated_qudits = self.compute_finished_blocks(
                    None, qudits, active_blocks,
                    finished_blocks, block_id, qudit_dependencies, cycle, num_qudits,  # noqa
                )
                block = {qudit: [cycle, cycle] for qudit in qudits}
                block[-1] = set()
                active_blocks.append(block)


            # Where the active qudit cycles keep getting updated
            while updated_qudits:

                # Check if any blocks corresponding to updated qudits
                # are eligible to be added to the circuit. If eligible,
                # update actives, dependencies, and updated qudits.
                final_regions = []
                new_updated_qudits = set()
                for qudit in updated_qudits:
                    blk_ids = list(qudit_dependencies[qudit].keys())
                    for blk_id in blk_ids:
                        num_passed = 0
                        for qdt, bounds in finished_blocks[blk_id].items():
                            if len(qudit_actives[qdt]) == 0:
                                num_passed += 1
                            elif next(iter(qudit_actives[qdt])) == bounds[0]:
                                num_passed += 1
                        if num_passed == len(finished_blocks[blk_id]):
                            for qdt, bounds in finished_blocks[blk_id].items():
                                for cycle in range(bounds[0], bounds[1] + 1):
                                    if cycle in qudit_actives[qdt]:
                                        del qudit_actives[qdt][cycle]
                                del qudit_dependencies[qdt][blk_id]
                                new_updated_qudits.add(qdt)
                            final_regions.append(
                                CircuitRegion(
                                {qdt: (bounds[0], bounds[1]) for qdt, bounds in finished_blocks[blk_id].items()},  # noqa
                                ),
                            )
                            del finished_blocks[blk_id]

                # If there are any regions
                if final_regions:
                    new_regions = self.merge_blocks(final_regions, circuit)
                    total_new_regions += len(new_regions)
                    final_regions.extend(new_regions)
                    # Sort the regions if multiple exist
                    if len(final_regions) > 1:
                        final_regions = self.topo_sort(final_regions)

                    total_regions += len(final_regions)
                    # Fold the final regions into a partitioned circuit
                    for region in final_regions:
                        region = circuit.downsize_region(region)
                        cgc = circuit.get_slice(region.points)
                        partitioned_circuit.append_gate(
                            CircuitGate(
                                cgc, True,
                            ), sorted(
                                list(
                                    region.keys(),
                                ),
                            ), list(
                                cgc.params,
                            ),
                        )

                updated_qudits = new_updated_qudits

        # Convert all remaining finished blocks and active blocks
        # into circuit regions
        final_regions = []
        for block in finished_blocks.values():
            final_regions.append(
                CircuitRegion(
                {qdt: (bounds[0], bounds[1]) for qdt, bounds in block.items()},  # noqa
                ),
            )
        for block in active_blocks:
            del block[-1]
            final_regions.append(CircuitRegion({qdt: (bounds[0], bounds[1]) for qdt, bounds in block.items()}))


        new_regions = self.merge_blocks(final_regions, circuit)

        final_regions.extend(new_regions)

        total_new_regions += len(new_regions)

        print(total_new_regions)

        # If there are any regions
        if final_regions:

            # Sort the regions if multiple exist
            if len(final_regions) > 1:
                final_regions = self.topo_sort(final_regions)

            total_regions += len(final_regions)

            # Fold the final regions into a partitioned circuit
            for region in final_regions:
                region = circuit.downsize_region(region)
                cgc = circuit.get_slice(region.points)
                partitioned_circuit.append_gate(
                    CircuitGate(
                        cgc, True,
                    ), sorted(
                        list(
                            region.keys(),
                        ),
                    ), list(
                        cgc.params,
                    ),
                )

        print(total_new_regions)
        print(final_regions)

        # Copy the partitioned circuit to the original circuit
        circuit.become(partitioned_circuit)

    def create_graph(self, regions):
        # Number of regions in the circuit
        num_regions = len(regions)

        in_edges = [set() for _ in range(num_regions)]
        out_edges = [set() for _ in range(num_regions)]
        edges = set()
        for i in range(num_regions-1):
            for j in range(i+1, num_regions):
                dependency = regions[i].dependency(regions[j])
                if dependency == 1:
                    in_edges[i].add(j)
                    out_edges[j].add(i)
                    edges.add((j,i))
                elif dependency == -1:
                    in_edges[j].add(i)
                    out_edges[i].add(j)
                    edges.add((i,j))

        return edges, in_edges, out_edges


    def try_to_merge_groups(self, groups: List[List[int]], reg_id: int, regions: List[CircuitRegion]) -> List[CircuitRegion]:
        region = regions[reg_id]
        new_regions = []
        for group in groups:
            # Loop through topo sorted blocks
            # If block contains all qubits in group, then we can merge!
            # If block contains only some qubits, then we can not merge and must split!
            merged = False
            # Remove qubits from current region and
            # Collect bounds
            bounds = region.remove_qubits(group)
            new_region = CircuitRegion(bounds)
            for i in range(reg_id + 1, len(regions)):
                next_reg = regions[i]
                can_merge = next_reg.has_all_qubits(group)
                if can_merge == 1:
                    # Merge in new region
                    next_reg.union(new_region)
                    merged = True
                    break
                elif can_merge == -1:
                    # Break out of search and split
                    break
                else:
                    # continue search
                    continue
            if not merged:
                # Split! Add new region to list
                new_regions.append(new_region)

        return new_regions


    def merge_blocks(self, regions: List[CircuitRegion], circuit: Circuit) -> List[CircuitRegion]:
        # Merge adjacent blocks so that all qubits are used at least once in a 2 qubit gate.
        # If not possible, separate out qubit into its own block
        regions = self.topo_sort(regions)
        new_regions = []
        for reg_id, region in enumerate(regions):
            groups = self.contains_separate_groups(region, circuit)
            if len(groups) > 1:
                # Try to merge last n - 1 groups
                add_regions = self.try_to_merge_groups(groups[1:], reg_id, regions)
                new_regions.extend(add_regions)

        return new_regions

    def contains_separate_groups(self, region: CircuitRegion, circuit: Circuit) -> list[int]:
        """Returns a list of lists describing all separate qubit groups"""
        ops = circuit.get_operations(region.points)
        op_locations = [(op.location[0], op.location[1]) for op in ops if len(op.location) == 2]
        G = nx.Graph(op_locations)
        groups = nx.connected_components(G)
        # Return groups with longest group first, try to merge smaller groups
        return sorted([tuple(c) for c in groups], key=len, reverse=True)




    def compute_finished_blocks(self, block, qudits, active_blocks, finished_blocks,
                                block_id, qudit_dependencies, cycle, num_qudits):
        """
        Add blocks with all inactive qudits to the finished_blocks list and
        remove them from the active_blocks list.

        """

        # Compile the qudits from the new operation,
        # the active qudits of the block being updated,
        # and the qudits in the block's inadmissible list
        qudits = set(qudits)
        if block:
            qudits.update([qudit for qudit in block if qudit != -1])
            qudits.update(block[-1])

        remove_blocks = []

        # For all active blocks
        for active_block in active_blocks:

            # If the active block is different than the block being updated
            if active_block != block:

                # If any of the qudits are in the active block or its
                # inadmissible list, then add those qudits to the
                # inadmissible list of the active block
                if any([
                    qudit in active_block or qudit in active_block[-1]
                    for qudit in qudits
                ]):
                    active_block[-1].update(qudits)

                # If the active block has reached its maximum size
                # and/or all of its qudits are inadmissible,
                # then add it to the remove list
                if (
                    len(active_block) - 1 == self.block_size and  # noqa
                    all([
                        qudit in active_block[-1]
                        for qudit in active_block if qudit != -1
                    ])
                ) or (
                    cycle - max(
                        active_block[qudit][1]
                        for qudit in active_block
                        if qudit != -1
                    ) > 200
                ) or len(active_block[-1]) == num_qudits:
                    remove_blocks.append(active_block)

        # Remove all blocks in the remove list from the active list
        # and add them to the finished blocks list after deleting
        # their inadmissible list and update qudit dependencies
        updated_qudits = set()
        for remove_block in remove_blocks:
            del remove_block[-1]
            finished_blocks[block_id] = remove_block
            for qudit in remove_block:
                qudit_dependencies[qudit][block_id] = None
                updated_qudits.add(qudit)
            active_blocks.remove(remove_block)
            block_id += 1

        return block_id, updated_qudits

    def topo_sort(self, regions):  # type: ignore
        """Topologically sort circuit regions."""

        edges, in_edges, out_edges = self.create_graph(regions)

        num_regions = len(regions)

        # Convert the list of number of in edges in to a min-heap
        in_edges = [[len(in_edges), i] for i, in_edges in enumerate(in_edges)]
        heapq.heapify(in_edges)

        index = 0
        sorted_regions = []

        self.edges.extend(edges)

        # While there are regions remaining to be sorted
        while index < num_regions:

            # Select the regions with zero remaining in edges
            selections = []
            while in_edges and not in_edges[0][0]:
                selections.append(heapq.heappop(in_edges))

            if not selections:
                raise RuntimeError('Unable to topologically sort regions.')

            # Add the regions to the sorted list
            for region in selections:
                sorted_regions.append(regions[region[1]])
                index += 1

            # Remove the regions from all other regions' in edges counts
            for i in range(len(in_edges)):
                in_edges[i][0] -= sum(
                    in_edges[i][1] in out_edges[region[1]]
                    for region in selections
                )

            # Convert in edges into a min-heap
            heapq.heapify(in_edges)

        return sorted_regions


    def create_networkx_graph(self):
        import networkx as nx
        graph = nx.DiGraph(self.edges)
        return graph