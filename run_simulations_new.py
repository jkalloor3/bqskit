from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import CNOTGate, CircuitGate
from sys import argv
import numpy as np
import pickle
import seaborn as sns
# import pandas as pd
from itertools import chain

from bqskit.ir.gates.parameterized.u3 import U3Gate
from bqskit.ir.point import CircuitPoint

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector

from util import load_circuit, get_circ_block_dirs, load_compiled_block_circuits_qp_inds, load_block, load_compiled_block_circuits
from util.distance import normalized_gp_frob_cost
from util.fix_global_phase import fix_phase

from bqskit.ext import bqskit_to_qiskit

shots = 100
partitioned_circ_save_file = "/pscratch/sd/j/jkalloor/bqskit/partitioned_circs/{circ_name}.pickle"

def generate_full_circuits(block_ensembles: dict, block_probs: list, block_names: list, pcirc: Circuit, ens_size: int) -> list[Circuit]:
    # Generate ens_size random circuits
    block_inds: list[list[int]] = []
    for block_name, probs in block_probs.items():
        rand_inds = np.random.choice(len(block_ensembles[block_name]), ens_size, p=probs)
        if ens_size == 1:
            rand_inds = [rand_inds[0]]
        block_inds.append(rand_inds)

    
    # print("BLock Indices: ", block_inds)
    # Now we have a list of block_inds, we need to generate the circuits
    all_circs = []
    for i in range(ens_size):
        circ = pcirc.copy()
        ind = 0
        for cycle, op in circ.operations_with_cycles():
            pt = CircuitPoint(cycle, op.location[0])
            block_name = block_names[ind]
            rand_inds = block_inds[ind]
            ind += 1
            assert isinstance(op.gate, CircuitGate)
            assert isinstance(op.gate._circuit, Circuit)
            new_block_circ = block_ensembles[block_name][rand_inds[i]]
            assert isinstance(new_block_circ, Circuit)
            assert op.gate._circuit.num_qudits == new_block_circ.num_qudits
            # print(block_name, op.gate._circuit.num_qudits)
            # dist = normalized_gp_frob_cost(new_block_circ.get_unitary(), op.get_unitary())
            # print("Distance: ", dist)
            circ.replace_with_circuit(pt, new_block_circ, as_circuit_gate=True)
            # op.gate._circuit = block_ensembles[block_name][rand_inds[i]]
            # op.params = block_ensembles[block_name][rand_inds[i]].params
            # op._num_params = len(op.params)
        # print(circ.get_unitary())
        # circ.unfold_all()
        all_circs.append(circ)
    return all_circs



# Circ 
if __name__ == '__main__':
    block_ensembles = {}
    block_probs = {}
    block_names = []
    circ_type = argv[1]

    np.set_printoptions(precision=2, threshold=np.inf, linewidth=np.inf)


    circ_name = argv[1]
    max_tol = float(argv[2]) if len(argv) > 2 else 0.01

    initial_circ = load_circuit(circ_name, opt=False)
    print("Original CX Count: ", initial_circ.count(CNOTGate()))
    if initial_circ.num_qudits <= 10:
        target = initial_circ.get_unitary()

    block_targets = {}
    block_dirs = get_circ_block_dirs(circ_name, max_tol, False)
    block_names = sorted([x[0] for x in block_dirs])
    print("Block dirs: ", block_dirs)
    print("Block names: ", block_names)

    # Get all the circuits
    for block_name, block_data in block_dirs:
        bc_file = load_block(circ_name, block_name)
        block_target = Circuit.from_file(bc_file) 
        block_un = block_target.get_unitary()
        block_targets[block_name] = block_un
        if block_data[0] == 'orig':
            block_ensembles[block_name] = [block_target]
            block_probs[block_name] = [1]
        elif block_data[0] == "_opt3":
            circ_file = load_block(circ_name, block_name, extra="_opt3")
            block_ensembles[block_name] = [Circuit.from_file(circ_file)]
            block_probs[block_name] = [1]
        else:
            print(block_name, block_data)
            inds, probs = load_compiled_block_circuits_qp_inds(*block_data)
            circuits = load_compiled_block_circuits(*block_data)
            ensemble = [circuits[i] for i in inds]
            block_ensembles[block_name] = ensemble
            block_probs[block_name] = probs



    # print("Num Circuits: ", [len(ens) for ens in block_ensembles.values()], flush=True)

    # print("LOADED CIRCUITS", flush=True)

    pcirc_file = partitioned_circ_save_file.format(circ_name=circ_name)
    partitioned_circ: Circuit = pickle.load(open(pcirc_file, 'rb'))
    # print("Partitioned Circuit: ", partitioned_circ.gate_counts)
    # print(partitioned_circ_save_file)

    ensemble_sizes = [1, 10, 20, 40, 80, 160]

    ensemble_circuits = []
    num_trials = 10
    ensemble_costs = []

    # Get avg cost
    target = initial_circ.get_unitary()
    for ens_size in ensemble_sizes:
        mean_costs = []
        print("Ensemble Size: ", ens_size, flush=True)
        for i in range(num_trials):
            # print("Trial: ", i, flush=True)
            ens = generate_full_circuits(block_ensembles, block_probs, block_names, partitioned_circ, ens_size)
            [fix_phase(c, target) for c in ens]
            mean_un = np.mean(np.array([c.get_unitary() for c in ens]), axis=0)
            mean_costs.append(normalized_gp_frob_cost(mean_un, target))
        ensemble_costs.append(mean_costs)
        print("Avg Cost: ", np.mean(mean_costs))
    

    # Plot ensemble_costs vs ensemble_sizes with mean and error bars
    import matplotlib.pyplot as plt
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    ensemble_sizes = np.array(ensemble_sizes)
    ensemble_costs = np.array(ensemble_costs)
    ensemble_costs = np.mean(ensemble_costs, axis=1)
    ensemble_costs_std = np.std(ensemble_costs, axis=1)
    plt.errorbar(ensemble_sizes, ensemble_costs, yerr=ensemble_costs_std, fmt='o', capsize=5)
    plt.xscale('log')
    plt.xlabel('Ensemble Size')
    plt.ylabel('Average Cost')
    plt.title(f'Average Cost vs Ensemble Size for {circ_name}')
    plt.grid()
    plt.savefig(f"ensemble_costs_{circ_name}.png")
    # plt.show()
