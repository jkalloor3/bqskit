"""This module implements the InstantiateCount pass"""
from __future__ import annotations

import logging
import numpy as np
from typing import Any

from bqskit.qis import UnitaryMatrix
from bqskit.ir import Gate, Circuit
from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.ir.gates import GlobalPhaseGate
from bqskit.runtime import get_runtime

from bqskit.ir.opt.cost.functions import NormalizedFrobeniusCostGenerator
from bqskit.ir.opt.cost.functions import HilbertSchmidtResidualsGenerator

hs_cost = HilbertSchmidtResidualsGenerator()
frob_cost = NormalizedFrobeniusCostGenerator()

class FixGlobalPhasePass(BasePass):
    
    def __init__(self):
        super().__init__()
        self.target = None

    def fix_phase(self, circuit: Circuit) -> tuple[float, float]:
        unitary = circuit.get_unitary()
        global_phase_correction = self.target.get_target_correction_factor(unitary)
        old_cost = frob_cost.calc_cost(circuit, self.target)
        circuit.append_gate(GlobalPhaseGate(1, global_phase=global_phase_correction), (0,))
        new_cost = frob_cost.calc_cost(circuit, self.target)
        return old_cost, new_cost

    async def run(
            self, 
            circuit : Circuit, 
            data: PassData
    ) -> None:
        self.target = data.target

        if "ensemble" in data:
            print("Number of Ensembles: ", len(data["ensemble"]), flush=True)
            for ens in data["ensemble"]:
                print("Fixing phase for ensemble with length: ", len(ens), flush=True)
                costs = [self.fix_phase(c) for c in ens]
                # costs = await get_runtime().map(
                #     self.fix_phase,
                #     ens,
                #     target=target
                # )
                old_costs = [c[0] for c in costs]
                new_costs = [c[1] for c in costs]
                print("Prev. avg. cost: ", np.mean(old_costs), flush=True)
                print("New avg. cost: ", np.mean(new_costs), flush=True)
            return
        
        # new_scan_sols = []
        # for psol in data["scan_sols"]:
        #     await self.fix_phase(psol[0], target)
        #     new = frob_cost.calc_cost(psol[0], target)
        #     new_scan_sols.append((psol[0], new))
        # data["scan_sols"] = new_scan_sols