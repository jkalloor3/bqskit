from bqskit.ir.circuit import Circuit
from sys import argv
from bqskit.exec.runners.quest import QuestRunner
from bqskit.exec.runners.sim import SimulationRunner
from bqskit import compile
import numpy as np
from bqskit.compiler.compiler import Compiler, WorkflowLike
from bqskit.ir.point import CircuitPoint
from bqskit.ir.gates import CNOTGate, GlobalPhaseGate, VariableUnitaryGate
# Generate a super ensemble for some error bounds
from bqskit.passes import *
from bqskit.runtime import get_runtime
import pickle
from bqskit.ir.opt.cost.functions import  HilbertSchmidtCostGenerator, FrobeniusNoPhaseCostGenerator
from bqskit.ir.opt.minimizers.lbfgs import LBFGSMinimizer
from bqskit.passes import ScanningGateRemovalPass


from util import SecondLEAPSynthesisPass, SubselectEnsemblePass, GenerateProbabilityPass, SelectFinalEnsemblePass, LEAPSynthesisPass2, QSearchSynthesisPass2

from bqskit import enable_logging

from pathlib import Path

import multiprocessing as mp

from os.path import join

from util import save_circuits, load_circuit, FixGlobalPhasePass, CalculateErrorBoundPass

enable_logging(True)

def get_distance(circ1: Circuit) -> float:
    global target
    return circ1.get_unitary().get_frobenius_distance(target)


def get_shortest_circuits(circ_name: str, tol: int, timestep: int,
                          num_unique_circs: int = 100) -> list[Circuit]:
    circ = load_circuit(circ_name)
    
    # workflow = gpu_workflow(tol, f"{circ_name}_{tol}_{timestep}")
    err_thresh = np.sqrt(10 ** (-1 * tol))
    extra_err_thresh = 3e-2 * err_thresh
    phase_generator = HilbertSchmidtCostGenerator()
    big_block_size = 8
    small_block_size = 3
    checkpoint_dir = f"fixed_block_checkpoints_min/{circ_name}_{timestep}_{tol}_{big_block_size}_{small_block_size}/"
    # proj_name = f"{circ_name}_{timestep}_{tol}_final"
    # base_checkpoint_dir = None
    # proj_name = None

    # layer_gen = SimpleLayerGenerator(single_qudit_gate_1=VariableUnitaryGate(1))

    fast_instantiation_options = {
        # 'multistarts': 1,
        'ftol': extra_err_thresh,
        'diff_tol_r': 1e-4,
        'max_iters': 10000,
        'min_iters': 100,
        # 'method': 'qfactor',
    }

    good_instantiation_options = {
        'multistarts': 8,
        'ftol': 5e-16,
        'gtol': 1e-15,
        'diff_tol_r': 1e-6,
        'max_iters': 100000,
        'min_iters': 1000,
    }

    slow_partitioner_passes = [
        ScanPartitioner(block_size=small_block_size),
        ScanPartitioner(block_size=big_block_size),
    ]

    fast_partitioner_passes = [
        QuickPartitioner(block_size=small_block_size),
        QuickPartitioner(block_size=big_block_size),
    ]

    if circ.num_qudits > 20:
        partitioner_passes = fast_partitioner_passes
        instantiation_options = fast_instantiation_options
    else:
        partitioner_passes = slow_partitioner_passes
        instantiation_options = good_instantiation_options


    synthesis_pass = LEAPSynthesisPass2(
        store_partial_solutions=True,
        # layer_generator=layer_gen,
        success_threshold = extra_err_thresh,
        partial_success_threshold=err_thresh,
        cost=phase_generator,
        instantiate_options=instantiation_options,
        use_calculated_error=True,
        max_psols=5
    )

    second_synthesis_pass = SecondLEAPSynthesisPass(
        success_threshold = extra_err_thresh,
        # layer_generator=layer_gen,
        partial_success_threshold=err_thresh,
        cost=phase_generator,
        instantiate_options=instantiation_options,
        use_calculated_error=True,
        max_psols=3
    )


    leap_workflow = [
        CheckpointRestartPass(checkpoint_dir, 
                                default_passes=partitioner_passes),
        ForEachBlockPass(
            [
                ForEachBlockPass(
                    [
                        synthesis_pass,
                        second_synthesis_pass,
                        FixGlobalPhasePass(),
                    ],
                    calculate_error_bound=False,
                    allocate_error=True,
                    allocate_error_gate=CNOTGate(),
                ),
                # ScanningGateRemovalPass(),
                CreateEnsemblePass(success_threshold=err_thresh, 
                                   use_calculated_error=True, 
                                   num_circs=num_unique_circs,
                                #    num_random_ensembles=5,
                                   cost=phase_generator, 
                                   solve_exact_dists=True),
                JiggleEnsemblePass(success_threshold=err_thresh, num_circs=5000, use_ensemble=True, cost=phase_generator),
                SubselectEnsemblePass(success_threshold=err_thresh, num_circs=100),
                GenerateProbabilityPass(size=50),
                UnfoldPass(),
            ],
            calculate_error_bound=True,
            error_cost_gen=phase_generator,
            allocate_error=True,
            allocate_error_gate=CNOTGate(),
        ),
        SelectFinalEnsemblePass(size=500)
    ]
    num_workers = 256
    compiler = Compiler(num_workers=num_workers)
    # target = circ.get_unitary()
    out_circ, data = compiler.compile(circ, workflow=leap_workflow, request_data=True)
    approx_circuits: list[Circuit] = data["final_ensemble"]
    print("Final Count: ", approx_circuits[0].count(CNOTGate()))
    print("Num Circs", len(approx_circuits))
    # actual_error = target.get_frobenius_distance(out_circ.get_unitary())
    # cost_error = generator.calc_cost(out_circ, target)
    # assert(np.allclose(actual_error, cost_error))
    return approx_circuits, data.error, out_circ.count(CNOTGate())

if __name__ == '__main__':
    global target
    circ_name = argv[1]
    timestep = int(argv[2])
    tol = int(argv[3])
    num_unique_circs = int(argv[4])
    circs, error_bound, count = get_shortest_circuits(circ_name, tol, timestep, num_unique_circs=num_unique_circs)
    sorted_circs = sorted(circs, key=lambda c: c.count(CNOTGate()))
    circ = load_circuit(circ_name)
    # target = circ.get_unitary()
    print("Error Bound", error_bound)
    # print("Actual Error", actual_error)
    print("Lowest Count", count)
    # print([c.count(CNOTGate()) for c in circs[:20]])
    # # print([c.get_unitary().get_frobenius_distance(circ.get_unitary()) for c in])
    # with mp.Pool() as pool:
    #     dists = pool.map(get_distance, circs)
    # dists = [get_distance(c) for c in circs]
    # print("Max Distance", max(dists))
    # print("Min Distance", min(dists))
    # print("Mean Distance", np.mean(dists))
    # print([c.get_unitary().get_distance_from(circ.get_unitary()) for c in circs[:20]])
    save_circuits(circs, circ_name, tol, timestep, ignore_timestep=True, extra_str=f"_{num_unique_circs}_circ_final_min")