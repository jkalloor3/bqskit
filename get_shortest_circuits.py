from bqskit.ir.circuit import Circuit
from sys import argv
from bqskit.exec.runners.quest import QuestRunner
from bqskit.exec.runners.sim import SimulationRunner
from bqskit import compile
import numpy as np
from bqskit.compiler.compiler import Compiler, WorkflowLike
from bqskit.ir.point import CircuitPoint
from bqskit.ir.gates import CNOTGate, GlobalPhaseGate
# Generate a super ensemble for some error bounds
from bqskit.passes import *
from bqskit.runtime import get_runtime
import pickle
from bqskit.ir.opt.cost.functions import HilbertSchmidtResidualsGenerator, HilbertSchmidtCostGenerator, FrobeniusCostGenerator
from bqskit.ir.opt.minimizers.lbfgs import LBFGSMinimizer
from qfactorjax.qfactor import QFactorJax

from util import AnalyzeBlockPass, SecondLEAPSynthesisPass

from bqskit import enable_logging

from pathlib import Path

import json

from os.path import join

from util import save_circuits, load_circuit, FixGlobalPhasePass


def gpu_workflow(tol: int, checkpoint_proj: str) -> WorkflowLike:
    gpu_block_size = 6
    num_multistarts = 32
    max_iters = 100000
    min_iters = 3
    diff_tol_r = 1e-5
    diff_tol_a = 0.0
    dist_tol = 10 ** (-tol)

    diff_tol_step_r = 0.1
    diff_tol_step = 200
    beta = 0

    batched_instantiation = QFactorJax(
        diff_tol_r=diff_tol_r,
        diff_tol_a=diff_tol_a,
        min_iters=min_iters,
        max_iters=max_iters,
        dist_tol=dist_tol,
        diff_tol_step_r=diff_tol_step_r,
        diff_tol_step=diff_tol_step,
        beta=beta,
    )
    instantiate_options = {
        'method': batched_instantiation,
        'multistarts': num_multistarts,
    }

    base_checkpoint_dir = "checkpoints"
    full_checkpoint_dir = join(base_checkpoint_dir, checkpoint_proj)
    gate_deletion_jax_pass = ScanningGateRemovalPass(instantiate_options=instantiate_options, checkpoint_proj=full_checkpoint_dir)
    # Check if checkpoint files exist
    workflow = [
        CheckpointRestartPass(base_checkpoint_dir, checkpoint_proj, 
                                default_passes=[
                                ToVariablePass(convert_all_single_qudit_gates=True),
                                QuickPartitioner(block_size=gpu_block_size),
                                ], save_as_qasm=False),
        ForEachBlockPass([
            gate_deletion_jax_pass,
        ]),
        UnfoldPass(),
        ToU3Pass(),
    ]

    return workflow


def get_shortest_circuits(circ_name: str, tol: int, timestep: int) -> list[Circuit]:
    circ = load_circuit(circ_name)
    
    # workflow = gpu_workflow(tol, f"{circ_name}_{tol}_{timestep}")

    extra_err_thresh = 1e-10
    err_thresh = 10 ** (-1 * tol) / 100
    generator = HilbertSchmidtCostGenerator()
    synthesis_pass = LEAPSynthesisPass(
        store_partial_solutions=True,
        success_threshold = extra_err_thresh,
        partial_success_threshold=err_thresh,
        cost=generator,
        instantiate_options={
            'min_iters': 100,
            'cost_fn_gen': generator,
            'method': 'minimization',
            'minimizer': LBFGSMinimizer()
        }
    )

    second_synthesis_pass = SecondLEAPSynthesisPass(
        success_threshold = extra_err_thresh,
        partial_success_threshold=err_thresh,
        cost=generator,
        instantiate_options={
            'min_iters': 100,
            'cost_fn_gen': generator,
            'method': 'minimization',
            'minimizer': LBFGSMinimizer()
        }
    )


    leap_workflow = [
        ToU3Pass(convert_all_single_qubit_gates=True),
        ScanPartitioner(3),
        ForEachBlockPass(
            [
                synthesis_pass,
                second_synthesis_pass,
                FixGlobalPhasePass()
            ]
        ),
        CreateEnsemblePass(success_threshold=err_thresh, num_circs=2000, cost=generator),
        JiggleEnsemblePass(success_threshold=err_thresh, num_circs=20000, use_ensemble=True),
    ]

    num_workers = 256
    compiler = Compiler(num_workers=num_workers)
    out_circ, data = compiler.compile(circ, workflow=leap_workflow, request_data=True)
    approx_circuits: list[Circuit] = data["ensemble"]
    print("Num Circs", len(approx_circuits))
    return approx_circuits

if __name__ == '__main__':
    circ_name = argv[1]
    timestep = int(argv[2])
    tol = int(argv[3])
    circs = get_shortest_circuits(circ_name, tol, timestep)
    sorted_circs = sorted(circs, key=lambda c: c.count(CNOTGate()))
    circ = load_circuit(circ_name)
    print(circ.count(CNOTGate()))
    print([c.count(CNOTGate()) for c in circs[:20]])
    print([c.get_unitary().get_frobenius_distance(circ.get_unitary()) for c in circs[:20]])
    print([c.get_unitary().get_distance_from(circ.get_unitary()) for c in circs[:20]])
    save_circuits(circs, circ_name, tol, timestep)