from bqskit.ir.circuit import Circuit
from sys import argv
from bqskit.compiler.compiler import Compiler
# Generate a super ensemble for some error bounds
from bqskit.passes import CheckpointRestartPass, ToU3Pass, ExtendBlockSizePass
from bqskit.passes import ForEachBlockPass, ScanPartitioner, QuickPartitioner
from util import JiggleEnsemblePass, CreateEnsemblePass
from ntro import NumericalTReductionPass
from bqskit import enable_logging
from util import LEAPSynthesisPass2, GenerateProbabilityPass, FixAnglesPass
from util import CheckEnsembleQualityPass, FixGlobalPhasePass, ConvertToZXZXZSimple
from util import load_circuit


def get_shortest_circuits(circ_name: str, tol: float, timestep: int,
                          num_unique_circs: int = 100, extra_str="") -> list[Circuit]:
    circ = load_circuit(circ_name, opt=opt)
    
    # workflow = gpu_workflow(tol, f"{circ_name}_{tol}_{timestep}")
    if tol == 0:
        err_thresh = 0.2
    else:
        err_thresh = 10 ** (-1 * tol)
    extra_err_thresh = 1e-2 * err_thresh
    big_block_size = 8
    small_block_size = 3
    checkpoint_dir = f"cliff_t_checkpoints/{circ_name}_{timestep}_{tol}_{big_block_size}_{small_block_size}/"
    
    slow_partitioner_passes = [
        ScanPartitioner(block_size=small_block_size),
        ExtendBlockSizePass(),
        ScanPartitioner(block_size=big_block_size),
    ]

    medium_partitioner_passes = [
        ScanPartitioner(block_size=big_block_size),
        ExtendBlockSizePass(),
        ForEachBlockPass(
            [
                ScanPartitioner(block_size=small_block_size),
            ]
        ),
    ]

    fast_partitioner_passes = [
        QuickPartitioner(block_size=small_block_size),
        ExtendBlockSizePass(),
        QuickPartitioner(block_size=big_block_size),
    ]

    fast_instantiation_options = {
        'multistarts': 2,
        'ftol': extra_err_thresh,
        'diff_tol_r': 1e-4,
        'max_iters': 10000,
        'min_iters': 100,
        'method': 'minimization',
    }

    good_instantiation_options = {
        'multistarts': 16,
        'ftol': 5e-16,
        'gtol': 1e-15,
        'diff_tol_r': 1e-6,
        'max_iters': 100000,
        'min_iters': 1000,
    }

    if circ.num_qudits > 20:
        partitioner_passes = fast_partitioner_passes
        instantiation_options = fast_instantiation_options
    elif circ.num_qudits > 10:
        partitioner_passes = medium_partitioner_passes
        instantiation_options = good_instantiation_options
    else:
        partitioner_passes = slow_partitioner_passes
        instantiation_options = good_instantiation_options

    create_ensemble_pass = CreateEnsemblePass(
            success_threshold=err_thresh, 
            use_calculated_error=False, 
            num_circs=num_unique_circs,
            num_random_ensembles=2,
            solve_exact_dists=True,
            sort_by_t=True,
            checkpoint_extra_str=""
    )

    synthesis_pass = LEAPSynthesisPass2(
        store_partial_solutions=True,
        success_threshold = extra_err_thresh,
        partial_success_threshold=err_thresh / 2,
        instantiate_options=instantiation_options,
        max_layer=14,
        max_psols=4
    )

    jiggle_pass = JiggleEnsemblePass(success_threshold=err_thresh, 
                                  num_circs=10000, 
                                  use_ensemble=True,
                                  use_calculated_error=False,
                                  checkpoint_extra_str="",
                                  count_t=True,
                                  flood_circ=False,
                                  do_u3_perturbation=True)

    num_workers = 128
    compiler = Compiler(num_workers=num_workers)
    leap_workflow = [
        FixAnglesPass(10),
        CheckpointRestartPass(checkpoint_dir, 
                                default_passes=partitioner_passes),
        ForEachBlockPass(
            [
                ForEachBlockPass(
                    [
                        synthesis_pass,
                        ConvertToZXZXZSimple(),
                        NumericalTReductionPass(
                            full_loops=5,
                            success_threshold=err_thresh / 10,
                            use_calculated_error=True),
                        ToU3Pass(ensemble=True, group=True),
                        FixGlobalPhasePass(),
                    ],
                    allocate_error=True,
                ),
                create_ensemble_pass,
                jiggle_pass,
                CheckEnsembleQualityPass(True, csv_name=""),
                GenerateProbabilityPass(),
            ]
        )
    ]
    compiler.compile(circ, workflow=leap_workflow)
    return

if __name__ == '__main__':
    global target
    circ_name = argv[1]
    timestep = int(argv[2])
    tol = int(argv[3])
    num_unique_circs = int(argv[4])
    opt = bool(int(argv[5])) if len(argv) > 5 else False
    opt_str = "_opt" if opt else ""
    print("OPT STR", opt_str, opt, flush=True)
    get_shortest_circuits(circ_name, tol, timestep, num_unique_circs=num_unique_circs, extra_str=opt_str)