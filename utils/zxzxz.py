"""This module implements the InstantiateCount pass"""
from __future__ import annotations

import pickle
from typing import Any
import numpy as np

import logging

from bqskit.ir import Circuit, Operation, CircuitPoint
from bqskit.qis import UnitaryMatrix
from bqskit.passes import PassAlias
from bqskit.passes.rules import ZXZXZDecomposition
from bqskit.passes.partitioning import GroupSingleQuditGatePass
from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData


from bqskit.ir.opt.cost.functions import  HilbertSchmidtResidualsGenerator
cost = HilbertSchmidtResidualsGenerator()

class ConvertToZXZXZSimple(BasePass):

    def __init__(self, group: bool = False) -> None:
        self.group = group


    @staticmethod
    def run_circuit(circuit: Circuit, group: bool = False) -> None:
        # For each CircuitGate, replace with correspond ZXZXZ
        for cycle, op in circuit.operations_with_cycles(reverse=True):
            if op.num_params >= 2:
                pt = CircuitPoint(cycle, op.location[0])
                new_circ = ZXZXZDecomposition.run_zxzxz_decomp_circ(op.get_unitary())
                # pts.append(CircuitPoint(cycle, op.location[0]))
                # new_ops.append(Operation(CircuitGate(new_circ), op.location))
                circuit.replace_with_circuit(pt, new_circ, as_circuit_gate=True)
        circuit.unfold_all()

    async def run(
            self, 
            circuit : Circuit, 
            data: PassData
    ) -> None:
        # For every circuit in data["scan_sols"], run the circuit
        leap_file = str(data["ensemble_name"]) + "_leap_" + str(data["index"]) + ".pkl"
        scan_sols: list[Circuit] = pickle.load(open(leap_file, "rb"))
        for circ in scan_sols:
            ConvertToZXZXZSimple.run_circuit(circ)

        pickle.dump(scan_sols, open(leap_file, "wb"))