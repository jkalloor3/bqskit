from bqskit.ir.circuit import Circuit
from sys import argv
from bqskit.compiler.compiler import Compiler
from bqskit.ir.gates import CNOTGate
# Generate a super ensemble for some error bounds
from bqskit.passes import LEAPSynthesisPass, CheckpointRestartPass
from bqskit.passes import ForEachBlockPass, ScanPartitioner, CreateEnsemblePass
from bqskit.passes import JiggleEnsemblePass
from ntro import NumericalTReductionPass
from bqskit import enable_logging
from util import normalized_frob_cost, LEAPSynthesisPass2, SecondLEAPSynthesisPass
from util import normalized_gp_frob_cost, EnsembleScanningGateRemovalPass
from util import CheckEnsembleQualityPass, FixGlobalPhasePass, ConvertToZXZXZSimple

# enable_logging(True)

def get_shortest_circuits(circ_name: str, circ_file: str, tol: int, num_unique_circs: int = 100) -> list[Circuit]:
    circ = Circuit.from_file(circ_file)
    print("Original CNOT Count: ", circ.count(CNOTGate()))
    
    # workflow = gpu_workflow(tol, f"{circ_name}_{tol}_{timestep}")
    if tol == 0:
        err_thresh = 0.2
    else:
        err_thresh = 10 ** (-1 * tol)

    extra_err_thresh = err_thresh * 0.01
    small_block_size = 3
    checkpoint_dir = f"block_checkpoints_clifft/{circ_name}_block2_{tol}/"

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
            num_random_ensembles=4,
            solve_exact_dists=True,
            sort_by_t=True,
            checkpoint_extra_str="_try1"
    )

    synthesis_pass = LEAPSynthesisPass2(
        store_partial_solutions=True,
        success_threshold = extra_err_thresh,
        partial_success_threshold=err_thresh,
        instantiate_options=instantiation_options,
        max_layer=14,
        max_psols=7
    )

    jiggle_pass = JiggleEnsemblePass(success_threshold=err_thresh, 
                                  num_circs=5000, 
                                  use_ensemble=True,
                                  use_calculated_error=False,
                                  checkpoint_extra_str="_try1",
                                  count_t=True)
    
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
                ConvertToZXZXZSimple(),
                NumericalTReductionPass(
                    full_loops=5,
                    success_threshold=err_thresh,
                    use_calculated_error=True),
                FixGlobalPhasePass(),
                scan_pass,
            ],
            allocate_error=True,
        ),
        create_ensemble_pass,
        jiggle_pass,
        CheckEnsembleQualityPass(False, csv_name="_try1"),
    ]
    num_workers = 128
    compiler = Compiler(num_workers=num_workers)
    # target = circ.get_unitary()
    out_circ, data = compiler.compile(circ, workflow=leap_workflow, request_data=True)
    # print("Initial Gate Counts: ", circ.gate_counts)
    # print("Final Gate Counts: ", out_circ.gate_counts)
    # final_dist = normalized_gp_frob_cost(out_circ.get_unitary(), circ.get_unitary())
    # print("Final Distance: ", final_dist)
    return

if __name__ == '__main__':
    circ_name = argv[1]
    block_num = argv[2]
    tol = int(argv[3])
    num_unique_circs = int(argv[4])
    circ_name = f"{circ_name}_{block_num}"
    circ_file = f"good_blocks/{circ_name}.qasm"
    # print("OPT STR", opt_str, opt, argv[5])
    get_shortest_circuits(circ_name, circ_file, tol, 
                          num_unique_circs=num_unique_circs)