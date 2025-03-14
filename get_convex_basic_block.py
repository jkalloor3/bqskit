from bqskit.ir.circuit import Circuit
from sys import argv
import glob
import os
from bqskit.compiler.compiler import Compiler, WorkflowLike
from bqskit.ir.gates import CNOTGate
# Generate a super ensemble for some error bounds
from bqskit.passes import CheckpointRestartPass, NOOPPass
from util import  ConvexHullFinderPass, CheckEnsembleQualityPass

good_instantiation_options = {
    'multistarts': 8,
    'ftol': 5e-16,
    'gtol': 1e-15,
    'diff_tol_r': 1e-6,
    'max_iters': 100000,
    'min_iters': 1000,
    'method': 'minimization'
}

base_checkpoint_dir = "/pscratch/sd/j/jkalloor/bqskit/convex_baseline/"
NUM_UNIQUE_CIRCS = 250

def get_ensemble_workflow(circ_name: str, tol: float) -> WorkflowLike:
    # workflow = gpu_workflow(tol, f"{circ_name}_{tol}_{timestep}")
    checkpoint_dir = f"{base_checkpoint_dir}/{circ_name}_{tol}/"
    err_thresh = 10 ** (-1 * tol)

    leap_workflow = [
        CheckpointRestartPass(checkpoint_dir, 
                                default_passes=[NOOPPass()]),
        ConvexHullFinderPass(100, err_thresh),
        CheckEnsembleQualityPass(False),
    ]
    return leap_workflow

def get_shortest_circuits(circ_data: list[tuple[str, str, float]], extra: str = "") -> list[Circuit]:
    '''
    Gets the corresponding workflow for the input
    
    Args:
        circ_data: list of tuples of the form (circ_name, circ_file, tol)
    '''
    workflows = [
        get_ensemble_workflow(circ_name, tol)
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

def find_file(circ_name: str, block_num: str, extra="") -> tuple[str, str]:
    circ_name = f"{circ_name}_{block_num}"

    circ_file = f"good_blocks{extra}/{circ_name}.qasm"
    if os.path.exists(circ_file):
        return circ_name, circ_file
    circ_file = f"bad_blocks{extra}/{circ_name}.qasm"
    if not os.path.exists(circ_file):
        raise Exception(f"File not found for {circ_name}: {block_num}")

    return circ_name, circ_file

def get_circ_data(circ_name: str, block_num: str | int, 
                  tol: float, extra: str = "") -> list[tuple[str, str, float]]:
    # Categorize circs into different categories and run them
    if tol == -1.0:
        tols = [0.5, 1.0, 3.0]
    else:
        tols = [tol]

    if block_num == "all_blocks":
        # Get all blocks
        good_circ_files = glob.glob(f"good_blocks{extra}/{circ_name}_*.qasm")
        # Ignore bad blocks for now
        bad_circ_files = glob.glob(f"bad_blocks{extra}/{circ_name}_*.qasm")
        # bad_circ_files = []
        all_circ_files = good_circ_files + bad_circ_files
        block_nums = [file.split('_')[-1].split('.')[0] for file in all_circ_files]
        circ_data = []
        for i, block_num in enumerate(block_nums):
            name, circ_file = find_file(circ_name, block_num, extra=extra)
            circ_file = all_circ_files[i]
            for tol in tols:
                circ_data.append((name, circ_file, tol))
        return circ_data
    else:
        circ_name, circ_file = find_file(circ_name, block_num, extra=extra)
        circ_data = []
        for tol in tols:
            circ_data.append((circ_name, circ_file, tol))
        return circ_data

if __name__ == '__main__':
    circ_name = argv[1]
    block_num = argv[2] if len(argv) > 2 else ""
    tol = float(argv[3]) if len(argv) > 3 else -1.0
    extra = argv[4] if len(argv) > 4 else "_tket"
    circ_data = get_circ_data(circ_name, block_num, tol, extra=extra)
    print(circ_data)
    get_shortest_circuits(circ_data, extra=extra)