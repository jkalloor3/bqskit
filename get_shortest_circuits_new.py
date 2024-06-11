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

import multiprocessing as mp

from os.path import join

from util import save_circuits, load_circuit, FixGlobalPhasePass, CalculateErrorBoundPass

# enable_logging(True)

def get_distance(circ1: Circuit) -> float:
    global target
    return circ1.get_unitary().get_frobenius_distance(target)

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


def get_shortest_circuits(circ_name: str, tol: int, timestep: int,
                          num_unique_circs: int = 100) -> list[Circuit]:
    circ = load_circuit(circ_name)
    
    # workflow = gpu_workflow(tol, f"{circ_name}_{tol}_{timestep}")

    extra_err_thresh = 1e-10
    err_thresh = 10 ** (-1 * tol)
    generator = HilbertSchmidtCostGenerator()

    base_checkpoint_dir = "checkpoints"
    proj_name = f"{circ_name}_{timestep}_{tol}"
    big_block_size = 8
    small_block_size = 3

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
        },
        use_calculated_error=True,
        checkpoint_dir=base_checkpoint_dir,
        checkpoint_proj=proj_name,
        append_block_id=True
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
        },
        checkpoint_dir=base_checkpoint_dir,
        checkpoint_proj=proj_name,
        append_block_id=True
    )

    leap_workflow = [
        CheckpointRestartPass(base_checkpoint_dir, proj_name, 
                                default_passes=[
                                QuickPartitioner(block_size=big_block_size),
                                ], save_as_qasm=False),
        ForEachBlockPass(
            [
                CheckpointRestartPass(base_checkpoint_dir, proj_name, 
                        default_passes=[
                        ScanPartitioner(block_size=small_block_size),
                        ], save_as_qasm=False,
                        append_block_id=True),
                ForEachBlockPass(
                    [
                        synthesis_pass,
                        second_synthesis_pass,
                    ],
                    calculate_error_bound=False,
                    allocate_error=True,
                    allocate_error_gate=CNOTGate(),
                ),
                CreateEnsemblePass(success_threshold=err_thresh, 
                                   use_calculated_error=True, 
                                   num_circs=num_unique_circs, 
                                   cost=generator, 
                                   solve_exact_dists=True),
                FixGlobalPhasePass(),
                UnfoldPass()
            ],
            calculate_error_bound=True,
            error_cost_gen=generator,
            allocate_error=True,
            allocate_error_gate=CNOTGate(),
        ),
        CreateEnsemblePass(success_threshold=err_thresh, num_circs=num_unique_circs, cost=generator, solve_exact_dists=False),
        JiggleEnsemblePass(success_threshold=err_thresh, num_circs=10000, use_ensemble=True),
        UnfoldPass()
    ]
    num_workers = 256
    compiler = Compiler(num_workers=num_workers)
    target = circ.get_unitary()
    out_circ, data = compiler.compile(circ, workflow=leap_workflow, request_data=True)
    approx_circuits: list[Circuit] = data["ensemble"]
    print("Num Circs", len(approx_circuits))
    actual_error = target.get_frobenius_distance(out_circ.get_unitary())
    cost_error = generator.calc_cost(out_circ, target)
    assert(np.allclose(actual_error, cost_error))
    return approx_circuits, data.error, actual_error, out_circ.count(CNOTGate())

if __name__ == '__main__':
    global target
    circ_name = argv[1]
    timestep = int(argv[2])
    tol = int(argv[3])
    num_unique_circs = int(argv[4])
    circs, error_bound, actual_error, count = get_shortest_circuits(circ_name, tol, timestep, num_unique_circs=num_unique_circs)
    sorted_circs = sorted(circs, key=lambda c: c.count(CNOTGate()))
    circ = load_circuit(circ_name)
    target = circ.get_unitary()
    print("Error Bound", error_bound)
    print("Actual Error", actual_error)
    print("Lowest Count", count)
    print([c.count(CNOTGate()) for c in circs[:20]])
    # print([c.get_unitary().get_frobenius_distance(circ.get_unitary()) for c in])
    with mp.Pool() as pool:
        dists = pool.map(get_distance, circs)
    # dists = [get_distance(c) for c in circs]
    print("Max Distance", max(dists))
    print("Min Distance", min(dists))
    print("Mean Distance", np.mean(dists))
    # print([c.get_unitary().get_distance_from(circ.get_unitary()) for c in circs[:20]])
    save_circuits(circs, circ_name, tol, timestep, ignore_timestep=True, extra_str=f"_{num_unique_circs}_circ")