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

    def run(self, circuit: Circuit, data: PassData) -> Any:
        data.error = self.cost.calc_cost(circuit, data.target)


class CalculateErrorBoundPass(BasePass):
    
    def __init__(self, block_size: int = 5) -> None:
        self.passes = [
            UnfoldPass(),
            QuickPartitioner(block_size=block_size),
            ForEachBlockPass(
                [
                    NOOPPass()
                ],
                calculate_error_bound=True
            )
        ]

    async def run(
            self, 
            circuit : Circuit, 
            data: PassData
    ) -> None:
        data_copy = data.copy()
        await Workflow(self.passes).run(circuit.copy(), data_copy)
        data.error = data_copy.error