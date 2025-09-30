from bqskit.ir.circuit import Circuit
from sys import argv
import glob
import os
from bqskit.compiler.machine import MachineModel
from bqskit.qis.graph import CouplingGraph
from bqskit.compiler.compiler import Compiler, WorkflowLike
from bqskit.ir.gates import CNOTGate
# Generate a super ensemble for some error bounds
from bqskit.passes import CheckpointRestartPass, SetModelPass

from bqskit.passes import ForEachBlockPass, ScanPartitioner
from util import JiggleEnsemblePass, CleanupBlockFiles
from util import  LEAPSynthesisPass2, SecondLEAPSynthesisPass
from util import CheckEnsembleQualityPass, FixGlobalPhasePass
from util import GenerateProbabilityPass
from util import CreateEnsemblePass

good_instantiation_options = {
    'multistarts': 8,
    'ftol': 5e-16,
    'gtol': 1e-15,
    'diff_tol_r': 1e-6,
    'max_iters': 100000,
    'min_iters': 1000,
    'method': 'minimization'
}

base_checkpoint_dir_form = "mapped_checkpoints_final_paper{extra}"
NUM_UNIQUE_CIRCS = 250

def get_ensemble_workflow(circ_name: str, tol: float, extra: str = "") -> WorkflowLike:
    # workflow = gpu_workflow(tol, f"{circ_name}_{tol}_{timestep}")
    base_checkpoint_dir = base_checkpoint_dir_form.format(extra=extra)
    checkpoint_dir = f"{base_checkpoint_dir}/{circ_name}_{tol}/"
    err_thresh = 10 ** (-1 * tol)

    extra_err_thresh = err_thresh * 0.01
    small_block_size = 3
    large_block_size = 8
    print("Checkpoint Dir: ", checkpoint_dir, flush=True)
    print("Error Threshold: ", err_thresh, flush=True)

    slow_partitioner_passes = [
        ScanPartitioner(block_size=small_block_size),
        ScanPartitioner(block_size=large_block_size),
    ]
    partitioner_passes = slow_partitioner_passes
    instantiation_options = good_instantiation_options

    create_ensemble_pass = CreateEnsemblePass(
            success_threshold=err_thresh, 
            use_calculated_error=False, 
            num_circs=NUM_UNIQUE_CIRCS,
            num_random_ensembles=2,
            solve_exact_dists=True,
    )

    synthesis_pass = LEAPSynthesisPass2(
        store_partial_solutions=True,
        success_threshold = extra_err_thresh,
        partial_success_threshold=err_thresh / 3,
        instantiate_options=instantiation_options,
        max_layer=14,
        max_psols=5
    )

    second_synthesis_pass = SecondLEAPSynthesisPass(
        success_threshold = extra_err_thresh,
        partial_success_threshold=err_thresh / 3,
        instantiate_options=instantiation_options,
        max_layer=14,
        max_psols=10
    )

    jiggle_pass = JiggleEnsemblePass(success_threshold=err_thresh, 
                                  num_circs=10000, 
                                  use_ensemble=True,
                                  use_calculated_error=False,
                                  jiggle_skew=0,
                                  do_u3_perturbation=True,
                                  flood_circ=True)
    
    # Map to mesh grid
    num_rows = 10
    num_cols = 10
    mesh = CouplingGraph.grid(num_rows, num_cols)
    model = MachineModel(100, coupling_graph=mesh)

    leap_workflow = [
        CheckpointRestartPass(checkpoint_dir, 
                                default_passes=partitioner_passes),
        SetModelPass(model),
        ForEachBlockPass(
            [
            ForEachBlockPass(
                [
                    synthesis_pass,
                    second_synthesis_pass,
                ],
                skip_file="ensemble_4_.qasms"
            ),
            create_ensemble_pass,
            jiggle_pass,
            CleanupBlockFiles(),
            CheckEnsembleQualityPass(False),
            GenerateProbabilityPass()
            ],
        )
    ]
    return leap_workflow

def get_shortest_circuits(circ_data: list[tuple[str, str, float]], extra: str = "") -> list[Circuit]:
    '''
    Gets the corresponding workflow for the input
    
    Args:
        circ_data: list of tuples of the form (circ_name, circ_file, tol)
    '''
    num_processes = len(circ_data)
    print(f"Num Processes: {num_processes}", flush=True)
    workflows = [
        get_ensemble_workflow(circ_name, tol, extra)
        for circ_name, _, tol in circ_data
    ]

    num_workers = min(os.cpu_count(), 200)
    compiler = Compiler(num_workers=num_workers)
    
    workflow_ind = 0
    ids: list[int] = []

    for _, circ_file, _ in circ_data:
        workflow = workflows[workflow_ind]
        workflow_ind += 1
        if workflow:
            circ = Circuit.from_file(circ_file)
            print("Original CNOT Count: ", circ.count(CNOTGate()), flush=True)
            ids.append(compiler.submit(circ, workflow))

    ind = 0
    for id in ids:
        compiler.result(id)
        print("Finished: ", circ_data[ind][0], flush=True)
        ind += 1
    compiler.close()
    print("Compiler Closed", flush=True)
    return

def find_file(circ_name: str) -> tuple[str, str]:
    folder_1 = f"/pscratch/sd/j/jkalloor/bqskit/ensemble_benchmarks"
    folder_2 = f"/pscratch/sd/j/jkalloor/bqskit/qce23_qfactor_benchmarks"

    file_1 = os.path.join(folder_1, f"{circ_name}.qasm")
    file_2 = os.path.join(folder_2, f"{circ_name}.qasm")
    if os.path.exists(file_1):
        circ_file = file_1
    elif os.path.exists(file_2):
        circ_file = file_2
    else:
        circ_file = ""
        print(f"File not found: {circ_name}", flush=True)
        return circ_name, circ_file


    return circ_name, circ_file


def get_circ_data(circ_name: str, block_num: str | int, 
                  tol: float, extra: str = "") -> list[tuple[str, str, float]]:
    # Categorize circs into different categories and run them
    if tol == -1.0:
        tols = [0.5, 1.0, 3.0]
    else:
        tols = [tol]

    circ_name, circ_file = find_file(circ_name)
    if circ_file == "":
        return []
    
    circ_data = []
    for tol in tols:
        circ_data.append((circ_name, circ_file, tol))
    return circ_data

if __name__ == '__main__':
    circ_name = argv[1]
    block_num = argv[2] if len(argv) > 2 else ""
    tol = float(argv[3]) if len(argv) > 3 else -1.0
    extra = argv[4] if len(argv) > 4 else ""
    circ_data = get_circ_data(circ_name, block_num, tol, extra=extra)
    print(circ_data)
    get_shortest_circuits(circ_data, extra=extra)