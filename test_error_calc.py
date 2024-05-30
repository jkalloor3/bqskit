from bqskit.ir.circuit import Circuit
from sys import argv
from bqskit.compiler.compiler import Compiler
from bqskit.ir.gates import CNOTGate, GlobalPhaseGate
# Generate a super ensemble for some error bounds
from bqskit.passes import *

from bqskit.ir.opt.cost.functions import HilbertSchmidtCostGenerator



from util import load_circuit, SecondLEAPSynthesisPass, FixGlobalPhasePass
from bqskit.ir.opt.minimizers.lbfgs import LBFGSMinimizer

import numpy as np



def run(circ_name: str, tol: int, timestep: int, skew: int) -> list[Circuit]:
    circ = load_circuit(circ_name, timestep)

    circ_orig = load_circuit("qc_binary_5q", timestep)

    err_thresh = 10 ** (-1 * tol)
    extra_err_thresh = 1e-10
    err_thresh = 10 ** (-1 * tol)
    generator = HilbertSchmidtCostGenerator()
    synthesis_pass = LEAPSynthesisPass(
        store_partial_solutions=True,
        success_threshold = extra_err_thresh,
        partial_success_threshold=err_thresh / 100,
        cost=generator,
        instantiate_options={
            'min_iters': 100,
            'cost_fn_gen': generator,
            'method': 'minimization',
            'minimizer': LBFGSMinimizer()
        },
        use_calculated_error=True
    )

    second_synthesis_pass = SecondLEAPSynthesisPass(
        success_threshold = extra_err_thresh,
        partial_success_threshold=err_thresh,
        use_calculated_error=True,
        cost=generator,
        instantiate_options={
            'min_iters': 100,
            'cost_fn_gen': generator,
            'method': 'minimization',
            'minimizer': LBFGSMinimizer()
        }
    )


    leap_workflow = [
        SetTargetPass(circ_orig.get_unitary()),
        ToU3Pass(convert_all_single_qubit_gates=True),
        ScanPartitioner(8),
        ForEachBlockPass(
            [
                ScanPartitioner(3),
                ForEachBlockPass(
                    [
                        synthesis_pass,
                        second_synthesis_pass,
                        # FixGlobalPhasePass(),
                        NOOPPass()
                    ],
                    calculate_error_bound=False,
                    allocate_error=True,
                    allocate_error_gate=CNOTGate(),
                    allocate_skew_factor=skew,
                    replace_filter="less-than-multi"
                ),
                UnfoldPass()
            ],
            calculate_error_bound=True,
            error_cost_gen=generator
        ),
        UnfoldPass()
    ]

    num_workers = 32
    compiler = Compiler(num_workers=num_workers)
    out_circ, data = compiler.compile(circ, workflow=leap_workflow, request_data=True)

    print(out_circ.gate_counts)
    print(circ.gate_counts)
    
    # assert(np.allclose(circ.params, out_circ.params))

    # assert(np.allclose(circ.get_unitary(), out_circ.get_unitary()))
    target = circ.get_unitary()
    out_utry = out_circ.get_unitary()
    global_phase_correction = target.get_target_correction_factor(out_utry)
    out_circ.append_gate(GlobalPhaseGate(1, global_phase=global_phase_correction), (0,))
    out_utry = out_circ.get_unitary()
    actual_error = target.get_frobenius_distance(out_utry)
    secondary_error = target.get_distance_from(out_utry)
    print("SECONDARY ERROR", secondary_error)
    assert(np.allclose(actual_error, generator.calc_cost(out_circ, target)))
    return data.error, out_circ.count(CNOTGate()), actual_error

if __name__ == '__main__':
    circ_name = argv[1]
    timestep = int(argv[2])
    tol = int(argv[3])
    for skew in range(1, 5):
        error, num_gates, actual_error = run(circ_name, tol, timestep, skew)

        print("SKEW", skew)
        print(f"Error Bound: {error}")
        print(f"Num Gates: {num_gates}")
        print(f"Actual Error: {actual_error}")