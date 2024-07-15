from bqskit.ir.circuit import Circuit
from sys import argv
from bqskit import compile
import numpy as np
from bqskit.compiler.compiler import Compiler, WorkflowLike
from bqskit.ir.gates import CNOTGate, GlobalPhaseGate, VariableUnitaryGate
# Generate a super ensemble for some error bounds
from bqskit.passes import *
from bqskit.ir.opt.cost.functions import HilbertSchmidtCostGenerator, FrobeniusNoPhaseCostGenerator
from bqskit.passes.search.heuristics import AStarHeuristic

from util import SecondQSearchSynthesisPass, SubselectEnsemblePass

from bqskit import enable_logging

import multiprocessing as mp

from os.path import join

from util import save_circuits, load_circuit, FixGlobalPhasePass

enable_logging(True)

def get_distance(circ1: Circuit) -> float:
    global target
    return circ1.get_unitary().get_frobenius_distance(target)

def get_shortest_circuits(circ_name: str, tol: int, timestep: int,
                          num_unique_circs: int = 100) -> list[Circuit]:
    circ = load_circuit(circ_name)
    
    # workflow = gpu_workflow(tol, f"{circ_name}_{tol}_{timestep}")
    err_thresh = 10 ** (-1 * tol)
    extra_err_thresh = 1e-2 * err_thresh
    generator_phase = FrobeniusNoPhaseCostGenerator()
    # generator = FrobeniusNoPhaseCostGenerator()
    generator = HilbertSchmidtCostGenerator()
    layer_gen = SimpleLayerGenerator(single_qudit_gate_1=VariableUnitaryGate(1))

    instantiation_options = {
        # 'multistarts': 1,
        'ftol': extra_err_thresh,
        'diff_tol_r': 1e-6,
        'max_iters': 100000,
        'min_iters': 100,
        'method': 'qfactor',
        # 'method': 'minimization',
        # 'minimizer': LBFGSMinimizer()
    }

    big_block_size = 8
    small_block_size = 3

    heuristic = AStarHeuristic(3, 1, cost_gen=generator)

    synthesis_pass = QSearchSynthesisPass(
        heuristic_function=heuristic,
        layer_generator=layer_gen,
        store_partial_solutions=True,
        success_threshold = extra_err_thresh,
        partial_success_threshold=err_thresh,
        cost=generator_phase,
        instantiate_options=instantiation_options,
        use_calculated_error=True
    )

    second_synthesis_pass = SecondQSearchSynthesisPass(
        heuristic_function=heuristic,
        layer_generator=layer_gen,
        success_threshold = extra_err_thresh,
        partial_success_threshold=err_thresh,
        cost=generator_phase,
        instantiate_options=instantiation_options,
        use_calculated_error=True
    )

    leap_workflow = [
        ToVariablePass(convert_all_single_qudit_gates=True),
        QuickPartitioner(block_size=big_block_size),
        ForEachBlockPass(
            [
                ScanPartitioner(block_size=small_block_size),
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
                CreateEnsemblePass(success_threshold=err_thresh, 
                                   use_calculated_error=True, 
                                   num_circs=num_unique_circs, 
                                   cost=generator_phase, 
                                   solve_exact_dists=True),
                JiggleEnsemblePass(success_threshold=err_thresh, num_circs=num_unique_circs * 50, use_ensemble=True, cost=generator, use_calculated_error=True),
                SubselectEnsemblePass(success_threshold=err_thresh, num_circs=num_unique_circs),
                UnfoldPass()
            ],
            calculate_error_bound=True,
            error_cost_gen=generator,
            allocate_error=True,
            allocate_error_gate=CNOTGate(),
        ),
        CreateEnsemblePass(success_threshold=err_thresh, num_circs=num_unique_circs, cost=generator, solve_exact_dists=False),
        JiggleEnsemblePass(success_threshold=err_thresh, num_circs=1000, use_ensemble=True, cost=generator),
        UnfoldPass()
    ]
    num_workers = 256
    compiler = Compiler(num_workers=num_workers)
    target = circ.get_unitary()
    out_circ, data = compiler.compile(circ, workflow=leap_workflow, request_data=True)
    approx_circuits: list[Circuit] = data["ensemble"]
    print("Num Circs", len(approx_circuits))
    actual_error = target.get_frobenius_distance(out_circ.get_unitary())
    print(actual_error)
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
    save_circuits(circs, circ_name, tol, timestep, ignore_timestep=True, extra_str=f"_{num_unique_circs}_circ_qsearch_subselect")