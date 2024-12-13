from bqskit.compiler.passdata import PassData
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
from bqskit.passes import UnfoldPass, LEAPSynthesisPass, CheckpointRestartPass
from bqskit.passes import ForEachBlockPass, ScanPartitioner, CreateEnsemblePass
from bqskit.passes import JiggleEnsemblePass
from bqskit import enable_logging
from util import normalized_frob_cost, LEAPSynthesisPass2, SecondLEAPSynthesisPass
from util import normalized_gp_frob_cost, EnsembleScanningGateRemovalPass
from util import CheckEnsembleQualityPass, FixGlobalPhasePass, JiggleScansPass

# enable_logging(True)

def get_shortest_circuits(circ_name: str, 
                          circ_file: str, 
                          tol: int, 
                          num_unique_circs: int = 100,
                          jiggle_skew: int =1,
                          ham_perturb: bool = False) -> list[Circuit]:
    circ = Circuit.from_file(circ_file)
    print("Original CNOT Count: ", circ.count(CNOTGate()))
    print("Jiggle Skew: ", jiggle_skew, flush=True)
    
    # workflow = gpu_workflow(tol, f"{circ_name}_{tol}_{timestep}")
    if tol == 0:
        err_thresh = 0.2
    else:
        err_thresh = 10 ** (-1 * tol)

    extra_err_thresh = err_thresh * 0.01
    small_block_size = 3
    checkpoint_dir = f"block_checkpoints_nisq_{jiggle_skew}/{circ_name}_{tol}_{num_unique_circs}/"

    good_instantiation_options = {
        'multistarts': 8,
        'ftol': 5e-16,
        'gtol': 1e-15,
        'diff_tol_r': 1e-6,
        'max_iters': 100000,
        'min_iters': 1000,
        'method': 'minimization'
    }

    slow_partitioner_passes = [
        ScanPartitioner(block_size=small_block_size),
        # ScanPartitioner(block_size=(small_block_size + 2)),
    ]
    partitioner_passes = slow_partitioner_passes
    instantiation_options = good_instantiation_options

    leap_pass = LEAPSynthesisPass(
        success_threshold=err_thresh,
        instantiate_options=instantiation_options,
    )

    create_ensemble_pass = CreateEnsemblePass(
            success_threshold=err_thresh, 
            use_calculated_error=False, 
            num_circs=num_unique_circs,
            num_random_ensembles=3,
            solve_exact_dists=True,
            checkpoint_extra_str="_try1"
    )

    synthesis_pass = LEAPSynthesisPass2(
        store_partial_solutions=True,
        success_threshold = extra_err_thresh,
        partial_success_threshold=err_thresh / 2,
        instantiate_options=instantiation_options,
        max_layer=14,
        max_psols=10
    )

    second_synthesis_pass = SecondLEAPSynthesisPass(
        success_threshold = extra_err_thresh,
        partial_success_threshold=err_thresh / 2,
        instantiate_options=instantiation_options,
        max_layer=14,
        max_psols=5
    )

    jiggle_pass = JiggleEnsemblePass(success_threshold=err_thresh, 
                                  num_circs=5000, 
                                  use_ensemble=True,
                                  use_calculated_error=False,
                                  checkpoint_extra_str="_try1",
                                  jiggle_skew=jiggle_skew,
                                  do_u3_perturbation=ham_perturb)
    
    scan_pass = EnsembleScanningGateRemovalPass(
        success_threshold=err_thresh,
        instantiate_options=instantiation_options,
    )

    leap_workflow = [
        CheckpointRestartPass(checkpoint_dir, 
                                default_passes=partitioner_passes),
        ForEachBlockPass(
            [
                synthesis_pass,
                # JiggleScansPass(success_threshold=err_thresh / 3),
                second_synthesis_pass,
                # scan_pass,
                FixGlobalPhasePass(),
            ],
            allocate_error=True,
        ),
        create_ensemble_pass,
        jiggle_pass,
        CheckEnsembleQualityPass(False, csv_name="_try1"),
    ]
    num_workers = 128
    compiler = Compiler(num_workers=num_workers)
    compiler.compile(circ, workflow=leap_workflow, request_data=True)
    return

if __name__ == '__main__':
    circ_name = argv[1]
    block_num = argv[2]
    tol = int(argv[3])
    num_unique_circs = int(argv[4])
    jiggle_skew = int(argv[5])
    ham_perturb = bool(int(argv[6])) if len(argv) > 6 else False
    circ_name = f"{circ_name}_{block_num}"
    circ_file = f"good_blocks/{circ_name}.qasm"
    # print("OPT STR", opt_str, opt, argv[5])
    get_shortest_circuits(circ_name, circ_file, tol, num_unique_circs=num_unique_circs, jiggle_skew=jiggle_skew, ham_perturb=ham_perturb)