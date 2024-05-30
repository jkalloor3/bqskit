"""This module implements the InstantiateCount pass"""
from __future__ import annotations

import logging
from typing import Any

from bqskit.ir import Circuit
from bqskit.passes import QuickPartitioner, ForEachBlockPass, UnfoldPass, NOOPPass
from bqskit.ir.opt.cost.functions import HilbertSchmidtResidualsGenerator
from bqskit.ir.opt.cost.generator import CostFunctionGenerator
from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.compiler.workflow import Workflow


class CalculateErrorPass(BasePass):
    def __init__(self,
        cost: CostFunctionGenerator = HilbertSchmidtResidualsGenerator()) -> None:
        self.cost = cost

    async def run(self, circuit: Circuit, data: PassData) -> Any:
        data.error = self.cost.calc_cost(circuit, data.target)


class CalculateErrorBoundPass(BasePass):
    
    def __init__(self, 
                 block_size: int = 5, 
                 cost: CostFunctionGenerator = HilbertSchmidtResidualsGenerator()) -> None:
        self.passes = [
            QuickPartitioner(block_size=block_size),
            ForEachBlockPass(
                [
                    CalculateErrorPass(cost=cost),
                ],
                calculate_error_bound=True
            )
        ]
        self.cost = cost

    async def run(
            self, 
            circuit : Circuit, 
            data: PassData
    ) -> None:
        data_copy = data.copy()
        circuit_copy = circuit.copy()
        circuit_copy.unfold_all()
        print("Full Cost", self.cost.calc_cost(circuit_copy, data.target))
        data_copy.error = 0
        await Workflow(self.passes).run(circuit_copy, data_copy)
        data.error = data_copy.error