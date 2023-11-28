from bqskit.ir.circuit import Circuit
from sys import argv
from bqskit.exec.runners.quest import QuestRunner
from bqskit.exec.runners.sim import SimulationRunner
from bqskit import compile
import numpy as np
from bqskit.compiler.compiler import Compiler
from bqskit.ir.point import CircuitPoint
# Generate a super ensemble for some error bounds
from bqskit.passes import *
from bqskit.runtime import get_runtime
import pickle

from pathlib import Path

def parse_data(
    circuit: Circuit,
    data: dict,
) -> tuple[list[list[tuple[Circuit, float]]], list[CircuitPoint]]:
    """Parse the data outputed from synthesis."""
    psols: list[list[tuple[Circuit, float]]] = []
    exact_block = circuit.copy()  # type: ignore  # noqa
    exact_block.set_params(circuit.params)
    exact_utry = exact_block.get_unitary()
    psols.append([(exact_block, 0.0)])

    for depth, psol_list in data['psols'].items():
        for psol in psol_list:
            dist = psol[0].get_unitary().get_distance_from(exact_utry)
            psols[-1].append((psol[0], dist))

    return psols



# Circ 
if __name__ == '__main__':
    circ_type = argv[1]

    if circ_type == "TFIM":
        target = np.loadtxt("/pscratch/sd/j/jkalloor/bqskit/ensemble_benchmarks/tfim_4-1.unitary", dtype=np.complex128)
    elif circ_type == "Heisenberg":
        target = np.loadtxt("/pscratch/sd/j/jkalloor/bqskit/ensemble_benchmarks/tfim_4-1.unitary", dtype=np.complex128)
    else:
        target = np.loadtxt("/pscratch/sd/j/jkalloor/bqskit/ensemble_benchmarks/qite_3.unitary", dtype=np.complex128)

    synth_circs = []

    err_thresh =1e-10

    initial_circ = Circuit.from_unitary(target)


    # workflow = [
    #     QFASTDecompositionPass(),
    #     ForEachBlockPass([LEAPSynthesisPass(), ScanningGateRemovalPass()]),
    #     UnfoldPass(),
    # ]

    compiler = Compiler()

    # circ = compiler.compile(initial_circ, workflow=workflow)

    # # Using Quest
    # quest_runner = QuestRunner(SimulationRunner(), compiler=compiler, sample_size=1, approx_threshold=1e-4)
    # approx_circuits = quest_runner.get_all_circuits(circ)


    # Just use LEAP
    # method = "leap"
    # synthesis_pass = LEAPSynthesisPass(
    #     store_partial_solutions=True,
    #     success_threshold = 1e-10,
    #     partial_success_threshold=1e-3
    # )
    
    # out_circ, data = compiler.compile(initial_circ, [synthesis_pass, 
    #                                                         CreateEnsemblePass(success_threshold=1e-3, 
    #                                                                            num_circs=500)], True)
    
    # approx_circuits: list[Circuit] = data["ensemble"]
    # # ensemble_dists = data["ensemble_dists"]

    # utries = [x.get_unitary() for x in approx_circuits]

    # dists = [x.get_distance_from(target) for x in utries]
    # dists = [x[1] for x in approx_circuits]



    # # data = data[ForEachBlockPass.key]
    # psols, pts = parse_data(blocked_circuit, data)

    # print(len(psols))


    # Use QFAST and Scanning Gate to get solutions
    method = "scan"
    workflow = [
        QFASTDecompositionPass(success_threshold=1e-12),
        ScanningGateRemovalPass(success_threshold=1e-3, store_all_solutions=True),
        CreateEnsemblePass(success_threshold=1e-3, num_circs=500)
    ]

    out_circ, data = compiler.compile(initial_circ, workflow=workflow, request_data=True)

    print("Finished Compiling!")
    approx_circuits: list[Circuit] = data["ensemble"]
    # ensemble_dists = data["ensemble_dists"]

    utries = [x.get_unitary() for x in approx_circuits]

    dists = [x.get_distance_from(target) for x in utries]
    # dists = [x[1] for x in approx_circuits]



    # Store approximate solutions
    dir = f"ensemble_approx_circuits/{method}/{circ_type}/"

    Path(dir).mkdir(parents=True, exist_ok=True)

    for i, circ in enumerate(approx_circuits):
        file = f"{dir}/circ_{i}.pickle"
        pickle.dump(circ, open(file, "wb"))

    print(len(approx_circuits))
    print(dists)




    # for seed in range(1, 500):
        # out_circ = compile(target, optimization_level=3, error_threshold=err_thresh, seed=seed)

        # if out_circ not in synth_circs:
        #     synth_circs.append(out_circ)