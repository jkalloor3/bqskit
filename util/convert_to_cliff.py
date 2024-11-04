"""This module implements the InstantiateCount pass"""
from __future__ import annotations

import pickle
from typing import Any

import logging

from bqskit.ir import Circuit, Operation, CircuitPoint
from bqskit.qis import UnitaryMatrix
from bqskit.passes import PassAlias
from bqskit.passes.rules import ZXZXZDecomposition
from bqskit.passes.partitioning import GroupSingleQuditGatePass
from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.ir.gates import U3Gate, RZGate, HGate, SqrtXGate, CircuitGate, RXGate, FixedRZGate, GlobalPhaseGate
from bqskit.utils.math import global_phase, correction_factor, canonical_unitary
from bqskit.runtime import get_runtime
from itertools import product
import numpy as np

import subprocess

from bqskit.ir.gates.constantgate import ConstantGate
from bqskit.ir.gates.qubitgate import QubitGate

from bqskit.ir.opt.cost.functions import  HilbertSchmidtResidualsGenerator
cost = HilbertSchmidtResidualsGenerator()

_logger = logging.getLogger(__name__)

class ConvertToZXZXZ(BasePass):

    def __init__(self, success_threshold: float = 1e-8, sub_epsilon: float = 1e-3, num_circs: int = 100, use_calculated_error: bool = True,
                 instantiation_options: dict[str, Any] = None):
        self.success_threshold = success_threshold
        self.sub_epsilon = sub_epsilon
        self.num_circs = num_circs
        self.use_calculated_error = use_calculated_error
        self.instantiation_options = instantiation_options

    def get_all_zxzxz_decomps(self) -> list[Circuit]:
        # Try to fix 1 angle in ZXZXZ decomposition
        circs = []
        for ind in [0,1,2]:
            for angle in np.linspace(0, np.pi*3/2, 4):
                circ = Circuit(1)
                for i in range(3):
                    if i == ind:
                        circ.append_gate(FixedRZGate(angle), (0,))
                    else:
                        circ.append_gate(RZGate(), (0,))
                    
                    if i != 2:
                        circ.append_gate(SqrtXGate(), (0,))

                circs.append(circ)

        return circs
    
    def get_default_circ(self, circuit: Circuit) -> Circuit:

        for cycle, op in circuit.operations_with_cycles(reverse=True):
            if isinstance(op.gate, RZGate):
                continue
            elif isinstance(op.gate, RXGate):
                circ = Circuit(1)
                circ.append_gate(HGate(), (0,))
                circ.append_gate(RZGate(), (0,), op.params)
                circ.append_gate(HGate(), (0,))
                assert cost.calc_cost(circ, op.get_unitary()) < 1e-10
                circuit.replace_with_circuit((cycle, op.location[0]), circ, as_circuit_gate=True)
            elif op.num_qudits == 1:
                # circ = ZXZXZDecomposition.get_zxzxz_circuit(op.get_unitary(), use_rx=False, use_u1=False, fix_global_phase=True)
                circ = Circuit(1)
                circ.append_gate(U3Gate(), (0,), U3Gate().calc_params(op.get_unitary()))
                # _logger.error("COST", cost.calc_cost(circ, op.get_unitary()), "Original Gate", op.gate, op.params)
                assert cost.calc_cost(circ, op.get_unitary()) < 1e-10
                circuit.replace_with_circuit((cycle, op.location[0]), circ, as_circuit_gate=True)
            else:
                continue
        circuit.unfold_all()
        return circuit

    async def get_zxzxz_decomp(self, target: UnitaryMatrix) -> Circuit:
        circs = self.get_all_zxzxz_decomps()
        inst_circs = await get_runtime().map(Circuit.instantiate, circs, target=target, **self.instantiation_options)

        default_circ = Circuit(1)
        for i in range(3):
            default_circ.append_gate(RZGate(), (0,))
            if i != 2:
                default_circ.append_gate(SqrtXGate(), (0,))

        best_circs = []
        dists = []
        for inst_circ in inst_circs:
            dist = cost.calc_cost(inst_circ, target)
            dists.append(dist)
            if dist < self.sub_epsilon:
                best_circs.append(inst_circ)

        best_circs.append(default_circ)

        return best_circs

    async def run(
            self, 
            circuit : Circuit, 
            data: PassData
    ) -> None:
        # Convert U3 gates to ZXZXZ
        pts = []
        targets = []
        locs = []
        
        if self.use_calculated_error:
            # print("OLD", self.success_threshold)
            success_threshold = self.success_threshold * data["error_percentage_allocated"]
            # print("NEW", self.success_threshold)
        else:
            success_threshold = self.success_threshold


        for cycle, op in circuit.operations_with_cycles(reverse=True):
            if isinstance(op.gate, U3Gate) or isinstance(op.gate, RXGate):
                targets.append(op.get_unitary())
                locs.append(op.location)
                pts.append((cycle, op.location[0]))

        if len(targets) == 0:
            # Nothing to do
            data["scan_sols"] = [(circuit.copy(), 0)]
            print("NO U3 GATES")
            return

        zxzxz_circs: list[list[Circuit]] = await get_runtime().map(self.get_zxzxz_decomp, targets)

        ensemble_circs = list(product(*zxzxz_circs))[:self.num_circs * 2]

        final_circs = []
        for circ_gates in ensemble_circs:
            cg_ops = [Operation(CircuitGate(zcirc), loc) for zcirc, loc in zip(circ_gates, locs)]
            circ = circuit.copy()
            circ.batch_replace(pts, cg_ops)
            circ.unfold_all()
            final_circs.append(circ)


        new_circs = await get_runtime().map(Circuit.instantiate, final_circs, target=data.target)

        dists = [cost.calc_cost(circ, data.target) for circ in new_circs]

        new_circs: list[tuple[Circuit, float]] = [(circ, dist) for circ, dist in zip(new_circs, dists) if dist < success_threshold ][:self.num_circs]

        if len(new_circs) == 0:
            print("NO SUCCESSFUL CIRCUITS")
            default_circ = self.get_default_circ(circuit.copy())
            assert cost.calc_cost(default_circ, data.target) < success_threshold
            new_circs = [(self.get_default_circ(circuit.copy()), 0)]

        tcount_approx = lambda circ: circ.num_params * 30
        counts = [tcount_approx(circ[0]) for circ in new_circs]

        print("Final Dists: ", dists)
        print("Final Counts: ", counts)
        # print("Final Scan Sols: ", new_circs)

        # for circ, _ in new_circs:
        #     print(circ.gate_counts)

        data["scan_sols"] = new_circs

        if "checkpoint_dir" in data:
            data["finished_zxzxz"] = True
            checkpoint_data_file = data["checkpoint_data_file"]
            pickle.dump(data, open(checkpoint_data_file, "wb"))
class ConvertToZXZXZSimple(BasePass):
    def run_circuit(self, circuit: Circuit) -> None:
        # Group Single Qudit Gates
        GroupSingleQuditGatePass.group(circuit)
        # For each CircuitGate, replace with correspond ZXZXZ
        cg_ops = []
        pts = []
        for cycle, op in circuit.operations_with_cycles(reverse=True):
            if isinstance(op.gate, CircuitGate):
                circ = op.gate._circuit
                circ = ZXZXZDecomposition.run_zxzxz_decomp(circ)
                cg_ops.append(Operation(CircuitGate(circ), op.location, circ.params))
                pts.append(CircuitPoint(cycle, op.location[0]))

        circuit.batch_replace(pts, cg_ops)
        # Unfold the circuit
        circuit.unfold_all()

    async def run(
            self, 
            circuit : Circuit, 
            data: PassData
    ) -> None:
        # For every circuit in data["scan_sols"], run the circuit
        scan_sols: list[tuple[Circuit, float]] = data["scan_sols"]
        for circ, _ in scan_sols:
            self.run_circuit(circ)
            # global_phase_correction = target.get_target_correction_factor(circ.get_unitary())
            # final_dist = cost.calc_cost(circ, data.target)
            # circ_copy = circ.copy()
            # circ_copy.append_gate(GlobalPhaseGate(1, global_phase=global_phase_correction), (0,))
            # corrected_dist = cost.calc_cost(circ_copy, data.target)
            # print("Global Phases Diff: ", global_phase_before - global_phase_after, "Dists: ", dist, final_dist, corrected_dist)