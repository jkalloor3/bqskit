from bqskit.ir.circuit import Circuit
from sys import argv
import glob
import os
import numpy as np
from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.compiler.compiler import Compiler, WorkflowLike
from bqskit.ir.gates import CNOTGate
# Generate a super ensemble for some error bounds
from bqskit.passes import CheckpointRestartPass
from bqskit.passes import ForEachBlockPass, ScanPartitioner, IfThenElsePass, PassPredicate
from util import JiggleEnsemblePass, FixGlobalPhasePass
from util import  LEAPSynthesisPass2, EnsScanningGateRemovalPass
from util import CheckEnsembleQualityPass
from util import GenerateProbabilityPass
from bqskit.passes import CheckpointRestartPass
from bqskit.passes import ForEachBlockPass, ScanPartitioner
from util import JiggleEnsemblePass
from ntro import NumericalTReductionPass
from util import LEAPSynthesisPass2, GenerateProbabilityPass, FixAnglesPass, UnFixTPass
from util import CheckEnsembleQualityPass, FixGlobalPhasePass, ConvertToZXZXZSimple

from util.distance import normalized_gp_frob_cost
from util import load_jiggled_ensemble, create_jiggled_unitaries

from bqskit.runtime import get_runtime
import itertools

class DoNothingPass(BasePass):

    async def run(self, circuit: Circuit, data: PassData) -> None:
        data['scan_sols'] = [(circuit.copy(), 0.0)]
        print("Do Nothing Pass, final distance: 0.0", flush=True)
        pass

class PrintDistancesPass(BasePass):

    def __init__(self, load_jiggles: bool = False):
        super().__init__()
        self.load_jiggles = load_jiggles

    async def run(self, circuit: Circuit, data: PassData) -> None:
        if self.load_jiggles:
            checkpoint_dir: str = data["checkpoint_dir"]
            print("Checkpoint Dir: ", checkpoint_dir, flush=True)
            ens_file = os.path.join(checkpoint_dir, "ensemble_0_.qasms")
            jiggle_file = os.path.join(checkpoint_dir, "ensemble_0_jiggles_.npy")
            cache_file = os.path.join(checkpoint_dir, "ensemble_0_cache_0.pkl")
            circ_params = load_jiggled_ensemble(ens_file, jiggle_file, cache_file)

            ensemble = await get_runtime().map(create_jiggled_unitaries, circ_params, 
                                    target=data.target, add_cost=True)
            ensemble = list(itertools.chain.from_iterable(ensemble))
            ds = [d for _, d in ensemble]
            actual_ds = [normalized_gp_frob_cost(u, data.target) for u, _ in ensemble]
        else:
            scan_sols = data.get('scan_sols', [])
            ds = [d for _, d in scan_sols]
            actual_ds = [normalized_gp_frob_cost(c.get_unitary(), data.target) for c, _ in scan_sols]
        print("Distances: ", ds, flush=True)
        print("Actual Distances: ", actual_ds, flush=True)
        pass 

class FilterDistancesPass(BasePass):

    def __init__(self, threshold: float = 0.001):
        super().__init__()
        self.threshold = threshold

    async def run(self, circuit: Circuit, data: PassData) -> None:
        scan_sols = data.get('scan_sols', [])
        new_scan_sols = [(c, d) for c, d in scan_sols if d < self.threshold]
        data['scan_sols'] = new_scan_sols
        pass 

class CountPredicate(PassPredicate):
    def get_truth_value(self, circuit, data):
        return circuit.count(CNOTGate()) < 26

good_instantiation_options = {
    'multistarts': 8,
    'ftol': 5e-16,
    'gtol': 1e-15,
    'diff_tol_r': 1e-6,
    'max_iters': 100000,
    'min_iters': 1000,
    'method': 'minimization'
}

SMALL_BLOCK_SIZE = 4
base_checkpoint_dir_form = "small_block_checkpoints_final_paper_{block_size}_clifft{extra}"
NUM_UNIQUE_CIRCS = 250

def get_ensemble_workflow(circ_name: str, tol: float, extra: str = "") -> WorkflowLike:
    # workflow = gpu_workflow(tol, f"{circ_name}_{tol}_{timestep}")
    ckpt_extra = extra
    base_checkpoint_dir = base_checkpoint_dir_form.format(block_size=SMALL_BLOCK_SIZE, 
                                                          extra=ckpt_extra)
    checkpoint_dir = f"{base_checkpoint_dir}/{circ_name}_{tol}/"
    err_thresh = 10 ** (-1 * tol) / 10

    extra_err_thresh = err_thresh * 0.01
    small_block_size = SMALL_BLOCK_SIZE
    print("Checkpoint Dir: ", checkpoint_dir, flush=True)
    print("Error Threshold: ", err_thresh, flush=True)

    # has_qft = "no_qft" in circ_name
    has_qft = False

    slow_partitioner_passes = [
        ScanPartitioner(block_size=small_block_size, ignore_qft=has_qft),
    ]
    partitioner_passes = slow_partitioner_passes
    instantiation_options = good_instantiation_options
    
    synthesis_pass = LEAPSynthesisPass2(
        store_partial_solutions=True,
        success_threshold = extra_err_thresh,
        partial_success_threshold=err_thresh / 5,
        max_layer_factor=1.01,
        instantiate_options=instantiation_options,
        max_layer=14,
        max_psols=20
    )
    
    full_success_threshold = err_thresh * 0.001
    success_threshold = err_thresh

    ntro = NumericalTReductionPass(
        full_loops=3,
        success_threshold=err_thresh * 5,
        use_calculated_error=True)

    jiggle_pass = JiggleEnsemblePass(success_threshold=err_thresh * 5, 
                                  num_circs=4000, 
                                  use_scan_sols=True,
                                  use_ensemble=False,
                                  use_calculated_error=False,
                                  jiggle_skew=0,
                                  count_t=True,
                                  do_u3_perturbation=True,
                                  flood_circ=False)

    leap_workflow = [
        FixAnglesPass(15),
        UnFixTPass(),
        CheckpointRestartPass(checkpoint_dir, 
                                default_passes=partitioner_passes),
        ForEachBlockPass(
            [
                # TketPass(),
                # IfThenElsePass(
                #     CountPredicate(),
                #     synthesis_pass,
                #     DoNothingPass()
                # ),
                # FixAnglesPass(int(tol) * 2 + 2, run_scan_sols=True),
                # ConvertToZXZXZSimple(group=False),
                # ntro,
                # FixAnglesPass(int(tol) * 2 + 2, run_scan_sols=True),
                # FixGlobalPhasePass(),
                # FilterDistancesPass(threshold=(err_thresh * 5)),
                # PrintDistancesPass(),
                jiggle_pass,
                # PrintDistancesPass(load_jiggles=True),
                # GenerateProbabilityPass(run_on_ensemble_0=True),
                # CheckEnsembleQualityPass(True),
            ]
        ),
    ]
    return leap_workflow


def check_if_finished(circ_name: str, tol: float, extra: str = "") -> tuple[bool, bool, str]:
    '''
    Given a circ_name and a tolerance, check if the ensemble is complete
    
    ret_1 -> completely finished
    ret_2 -> jiggle pass finished
    ret_3 -> extra_str
    
    '''
    ckpt_extra = extra
    base_checkpoint_dir = base_checkpoint_dir_form.format(block_size=SMALL_BLOCK_SIZE, 
                                                          extra=ckpt_extra)
    checkpoint_dir = os.path.join(base_checkpoint_dir, f"{circ_name}_{tol}")
    final_file = os.path.join(checkpoint_dir, "data.csv")
    if os.path.exists(final_file):
        return True, True, ""
    # Check if there is a jiggle .npy file in the checkpoint dir for at least 5
    jiggle_file = os.path.join(checkpoint_dir, "*4_jiggles*.npy")
    jiggle_files = glob.glob(jiggle_file)
    if len(jiggle_files) == 0:
        return False, False, ""
    # Try to get extra_str from jiggle files
    extra_str = jiggle_files[0].split("4_jiggles_")[1]
    # Remove everything after .
    extra_str = extra_str.split(".npy")[0]
    return False, True, extra_str

def get_final_workflow(circ_name: str, tol: float, extra: str = "") -> WorkflowLike | None:
    # Check if already finished
    # checkpoint_dir = f"{base_checkpoint_dir}/{circ_name}_{tol}/"
    ckpt_extra = extra
    base_checkpoint_dir = base_checkpoint_dir_form.format(block_size=SMALL_BLOCK_SIZE, 
                                                          extra=ckpt_extra)
    checkpoint_dir = os.path.join(base_checkpoint_dir, f"{circ_name}_{tol}")
    print(f"Checkpoint Dir: {checkpoint_dir}", flush=True)
    finished, jiggle_finished, _ = check_if_finished(circ_name, tol, extra=extra)
    if finished:
        print(f"Already finished {circ_name} {tol}", flush=True)
        return None
    if not jiggle_finished:
        print(f"Jiggle not finished {circ_name} {tol}", flush=True)
        return get_ensemble_workflow(circ_name, tol, extra=extra)
    print("Finding Final Ensemble for ", circ_name, flush=True)
    workflow = [
        CheckpointRestartPass(checkpoint_dir, 
                                default_passes=[]),
        CheckEnsembleQualityPass(False),
        # GenerateProbabilityPass()
    ]
    return workflow

def get_shortest_circuits(circ_data: list[tuple[str, str, float]], extra: str = "") -> list[Circuit]:
    '''
    Gets the corresponding workflow for the input
    
    Args:
        circ_data: list of tuples of the form (circ_name, circ_file, tol)
    '''
    workflows = [
        get_final_workflow(circ_name, tol, extra)
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
            ccount = circ.count(CNOTGate())
            if ccount > 2:
                print("Original CNOT Count: ", circ.count(CNOTGate()), 
                      flush=True)
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

    if extra != "_tket":
        extra = ""

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
        tols = [1.0, 2.0, 3.0, 4.0, 5.0]
    elif tol == -2.0:
        tols = [1.0, 2.0, 3.0]
    elif tol == -3.0:
        tols = [4.0, 5.0]
    else:
        tols = [tol]
    ckpt_extra = extra
    
    base_checkpoint_dir = base_checkpoint_dir_form.format(block_size=SMALL_BLOCK_SIZE, 
                                                          extra=ckpt_extra)
    if circ_name == "all_probs":
        # Get all the circ names, block_nums and tols which have a .npy file
        # but no ensemble_final.qasms file
        circ_data = []
        print("Finding all circ data", flush=True)
        for dir in os.listdir(base_checkpoint_dir):
            parts = dir.split("_")
            block_num = parts[-2]
            tol = float(parts[-1])
            circ_name = "_".join(parts[:-2])
            if circ_name.startswith("shor"):
                continue
            circ_name, circ_file = find_file(circ_name, block_num, extra=extra)
            for tol in tols:
                finished, jiggle_finished, _ = check_if_finished(circ_name, tol, extra=extra)
                if not finished and jiggle_finished:
                    circ_data.append((circ_name, circ_file, tol))

        if len(circ_data) > 40:
            circ_data = circ_data[:40]
            print("Limiting to 10 circs", flush=True)
        return circ_data
    
    else:
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

            # Only do 20 blocks
            if len(circ_data) > 100:
                rand_inds = np.random.choice(len(circ_data), 100, replace=False)
                circ_data = [circ_data[i] for i in rand_inds]
                print("Limiting to 100 blocks", flush=True)
            return circ_data
        else:
            circ_name, circ_file = find_file(circ_name, block_num, extra=extra)
            circ_data = []
            for tol in tols:
                circ_data.append((circ_name, circ_file, tol))
            return circ_data

if __name__ == '__main__':
    circ_name = argv[1]
    block_num = argv[2] if len(argv) > 2 else "all_blocks"
    tol = float(argv[3]) if len(argv) > 3 else -1.0
    extra = argv[5] if len(argv) > 5 else "_tket"
    circ_data = get_circ_data(circ_name, block_num, tol, extra=extra)
    np.random.seed(42)
    print(circ_data)
    get_shortest_circuits(circ_data, extra=extra)