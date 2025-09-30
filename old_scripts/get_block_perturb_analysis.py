from bqskit.ir.circuit import Circuit
from sys import argv
import glob
import os
import pickle
from pathlib import Path
from typing import Generator
from bqskit.runtime import get_runtime
from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from bqskit.compiler.compiler import Compiler, WorkflowLike
from bqskit.ir.gates import CNOTGate, U3Gate
# Generate a super ensemble for some error bounds
from bqskit.passes import CheckpointRestartPass
from bqskit.passes import ForEachBlockPass, ScanPartitioner
from bqskit.ir.lang.qasm2 import OPENQASM2Language
from util import  gg_gate_def, HamiltonianNoisePass
from itertools import product
from util import normalized_gp_frob_cost

def init_circuit(num_qubits: int) -> Circuit:
    circuit = Circuit(num_qubits)
    for q in range(num_qubits):
        circuit.append_gate(U3Gate(), q)
    return circuit


def apply_atomic_unit(circuit: Circuit, edge: tuple[int, int]) -> Circuit:
    circuit.append_gate(CNOTGate(), edge)
    circuit.append_gate(U3Gate(), edge[0])
    circuit.append_gate(U3Gate(), edge[1])
    return circuit


def edge_product(edges: list[tuple[int, int]], num_layers: int) -> Generator[tuple[int, int], None, None]:
    # Generator that yields the next edge product
    # Do not choose the same edge 3 times
    prev_edge_ind = -1
    all_sets = [([], prev_edge_ind, 0)]
    # Do a tree search 
    while len(all_sets) > 0:
        edge_set, prev_edge_ind, num_repeats = all_sets.pop()
        if len(edge_set) == num_layers:
            yield edge_set
            continue
        for i, edge in enumerate(edges):
            if i == prev_edge_ind and num_repeats == 2:
                continue
            else:
                new_edge_set = edge_set + [edge]
                if i == prev_edge_ind:
                    all_sets.append((new_edge_set, i, num_repeats + 1))
                else:
                    all_sets.append((new_edge_set, i, 1))
        
    return



def get_all_ansatze(n: int, num_gates: int) -> list[Circuit]:
    edges = [(i, j) for i, j in product(range(n), range(n)) if i < j]

    templates = []
    edges_per_circuit = edge_product(edges, num_gates)
    for edge_set in edges_per_circuit:
        circuit = init_circuit(n)
        for edge in edge_set:
            circuit = apply_atomic_unit(circuit, edge)
        templates.append(circuit)
    return templates


class AnalyzePerturbation(BasePass):

    def __init__(self, eps: float) -> None:
        self.eps = eps

    async def run(self, circuit: Circuit, data: PassData) -> None:
        # Calculate the number of ensembles from the previous pass
        orig_num_cnots = min(circuit.count(CNOTGate()), 8)

        # Get the perturbations
        perturbations = HamiltonianNoisePass.get_perturbation_bases(
            circuit.num_qudits,
            self.eps,
            circuit.num_qudits
        )

        new_targets = [data.target @ p for p in perturbations]

        # We want to track how many targets we can succesfully instantiate
        success_threshold = self.eps ** 2
        all_data = {}

        print("Orig CNOT Count: ", orig_num_cnots, flush=True)

        for i in range(orig_num_cnots):
            ansatzes = get_all_ansatze(circuit.num_qudits, i)
            # For each target, instantiate all the ansatze
            target_data = []
            for target in new_targets:
                out_circs = await get_runtime().map(Circuit.instantiate, ansatzes, target)
                dists = [normalized_gp_frob_cost(c.get_unitary(), target) for c in out_circs]
                # Add how many are below the threshold
                num_sols = sum([d < success_threshold for d in dists])
                target_data.append((min(dists), num_sols))
            all_data[i] = target_data

        # Save the data
        checkpoint_dir = data.get("checkpoint_dir")
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        save_file = os.path.join(checkpoint_dir, "perturbation_data.pkl")
        with open(save_file, "wb") as f:
            pickle.dump(all_data, f)

good_instantiation_options = {
    'multistarts': 8,
    'ftol': 5e-16,
    'gtol': 1e-15,
    'diff_tol_r': 1e-6,
    'max_iters': 100000,
    'min_iters': 1000,
    'method': 'minimization'
}

base_checkpoint_dir_form = "/pscratch/sd/j/jkalloor/bqskit/block_ens_perturb_analysis{extra}"
NUM_UNIQUE_CIRCS = 250
small_block_size = 3

def get_ensemble_workflow(circ_name: str, tol: float, extra: str = "") -> WorkflowLike:
    # workflow = gpu_workflow(tol, f"{circ_name}_{tol}_{timestep}")
    ckpt_extra = extra
    base_checkpoint_dir = base_checkpoint_dir_form.format(extra=ckpt_extra)
    checkpoint_dir = f"{base_checkpoint_dir}/{circ_name}_{tol}/"
    err_thresh = 10 ** (-1 * tol)

    slow_partitioner_passes = [
        ScanPartitioner(block_size=small_block_size),
    ]
    partitioner_passes = slow_partitioner_passes

    analyze_perturbation_pass = AnalyzePerturbation(err_thresh)

    leap_workflow = [
        CheckpointRestartPass(checkpoint_dir, 
                                default_passes=partitioner_passes),
        ForEachBlockPass(
            [
                analyze_perturbation_pass,
            ],
        ),
    ]
    return leap_workflow


def get_shortest_circuits(circ_data: list[tuple[str, str, float]], extra: str = "") -> list[Circuit]:
    '''
    Gets the corresponding workflow for the input
    
    Args:
        circ_data: list of tuples of the form (circ_name, circ_file, tol)
    '''
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
        tols = [3.0, 4.0, 5.0]
    else:
        tols = [tol]

    ckpt_extra = extra
    base_checkpoint_dir = base_checkpoint_dir_form.format(extra=ckpt_extra)
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
            circ_name, circ_file = find_file(circ_name, block_num, extra=extra)
            for tol in tols:
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
    circ_data = get_circ_data(circ_name, block_num, tol, extra="_tket")
    print(circ_data)
    get_shortest_circuits(circ_data, extra="_tket")
    # a_pass = AnalyzePerturbation(0.001)

    # base_circ = Circuit(3)
    # base_circ.append_gate(CNOTGate(), (0, 1))
    # base_circ.append_gate(CNOTGate(), (1, 2))
    # base_circ.append_gate(CNOTGate(), (0, 1))
    # base_circ.append_gate(CNOTGate(), (0, 1))

    # comp = Compiler(num_workers=30)
    # comp.compile(base_circ, [a_pass])
