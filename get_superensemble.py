from bqskit.ir.circuit import Circuit
from sys import argv
from bqskit.exec.runners.quest import QuestRunner
from bqskit.exec.runners.sim import SimulationRunner
from bqskit import compile
import numpy as np
from bqskit.compiler.compiler import Compiler
from bqskit.ir.point import CircuitPoint
from bqskit.ir.gates import CNOTGate
# Generate a super ensemble for some error bounds
from bqskit.passes import *
from bqskit.runtime import get_runtime
import pickle
from bqskit.ir.opt.cost.functions import HilbertSchmidtResidualsGenerator, HilbertSchmidtCostGenerator, FrobeniusCostGenerator
from bqskit.ir.opt.minimizers.lbfgs import LBFGSMinimizer
from bqskit.ir.opt.minimizers.scipy import ScipyMinimizer

from bqskit import enable_logging

from pathlib import Path

import json

# def parse_data(
#     circuit: Circuit,
#     data: dict,
# ) -> tuple[list[list[tuple[Circuit, float]]], list[CircuitPoint]]:
#     """Parse the data outputed from synthesis."""
#     psols: list[list[tuple[Circuit, float]]] = []
#     exact_block = circuit.copy()  # type: ignore  # noqa
#     exact_block.set_params(circuit.params)
#     exact_utry = exact_block.get_unitary()
#     psols.append([(exact_block, 0.0)])

#     for depth, psol_list in data['psols'].items():
#         for psol in psol_list:
#             dist = psol[0].get_unitary().get_distance_from(exact_utry)
#             psols[-1].append((psol[0], dist))

#     return psols

from bqskit.ir.opt.cost.differentiable import DifferentiableCostFunction
from bqskit.ir.opt.cost.generator import CostFunctionGenerator
from bqskit.qis.state.state import StateVector
from bqskit.qis.state.system import StateSystem

from bqskit.ir.circuit import Circuit
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.ir.opt.cost.function import CostFunction
from bqskit.qis.unitary.unitary import RealVector
import numpy.typing as npt

# Circ 
if __name__ == '__main__':
    # enable_logging(True)
    np.set_printoptions(precision=4, threshold=np.inf, linewidth=np.inf)
    circ_type = argv[1]
    timestep = int(argv[2])
    method = argv[3]
    tol = int(argv[4])

    if circ_type == "TFIM":
        initial_circ = Circuit.from_file(f"/pscratch/sd/j/jkalloor/bqskit/ensemble_benchmarks/TFIM_3_timesteps/TFIM_3_{timestep}.qasm")
        initial_circ.remove_all_measurements()
        # target = np.loadtxt("/pscratch/sd/j/jkalloor/bqskit/ensemble_benchmarks/tfim_4-1.unitary", dtype=np.complex128)
    elif circ_type == "Heisenberg":
        initial_circ = Circuit.from_file("/pscratch/sd/j/jkalloor/bqskit/ensemble_benchmarks/heisenberg_3.qasm")
    elif circ_type == "Heisenberg_7":
        initial_circ = Circuit.from_file("/pscratch/sd/j/jkalloor/bqskit/ensemble_benchmarks/heisenberg7.qasm")
    elif circ_type == "Hubbard":
        initial_circ = Circuit.from_file("/pscratch/sd/j/jkalloor/bqskit/ensemble_benchmarks/hubbard_4.qasm")
        # target = np.loadtxt("/pscratch/sd/j/jkalloor/bqskit/ensemble_benchmarks/tfim_4-1.unitary", dtype=np.complex128)
    else:
        target = np.loadtxt("/pscratch/sd/j/jkalloor/bqskit/ensemble_benchmarks/qite_3.unitary", dtype=np.complex128)
        initial_circ = Circuit.from_unitary(target)

    target = initial_circ.get_unitary()
    print(initial_circ)

    synth_circs = []

    # TODO: Divide by number of blocks yo
    err_thresh = 10 ** (-1 * tol)
    extra_err_thresh = 1e-15

    orig_depth = initial_circ.depth
    orig_count = initial_circ.count(CNOTGate())

    # workflow = [
    #     QFASTDecompositionPass(),
    #     ForEachBlockPass([LEAPSynthesisPass(), ScanningGateRemovalPass()]),
    #     UnfoldPass(),
    # ]

    compiler = Compiler(num_workers=-1)
    
    # circ = compiler.compile(initial_circ, workflow=workflow)

    # # Using Quest
    # quest_runner = QuestRunner(SimulationRunner(), compiler=compiler, sample_size=1, approx_threshold=1e-4)
    # approx_circuits = quest_runner.get_all_circuits(circ)

    approx_circuits: list[Circuit] = []

    generator = HilbertSchmidtCostGenerator()
    # generator = FrobeniusCostGenerator()

    # Just use LEAP
    if method == "leap":
        synthesis_pass = LEAPSynthesisPass(
            store_partial_solutions=True,
            success_threshold = extra_err_thresh,
            partial_success_threshold=err_thresh,
            cost=generator,
            instantiate_options={
                'min_iters': 100,
                'cost_fn_gen': generator,
                'method': 'minimization',
                'minimizer': LBFGSMinimizer()
            }
        )

        workflow = [
            ScanPartitioner(3),
            ForEachBlockPass([
                synthesis_pass
            ]),
            CreateEnsemblePass(success_threshold=err_thresh, num_circs=20000)
        ]

        old_workflow = [synthesis_pass, CreateEnsemblePass(success_threshold=err_thresh, num_circs=10000)]
        
        out_circ, data = compiler.compile(initial_circ, workflow, request_data=True)
        approx_circuits: list[Circuit] = data["ensemble"]
    elif method == "treescan":
        workflow = [
            TreeScanningGateRemovalPass(success_threshold=err_thresh, store_all_solutions=True, tree_depth=7),
            CreateEnsemblePass(success_threshold=err_thresh, num_circs=10000)
        ]

        out_circ, data = compiler.compile(initial_circ, workflow=workflow, request_data=True)
        approx_circuits: list[Circuit] = data["ensemble"]
    elif method == "quest":
        quest_runner = QuestRunner(SimulationRunner(), compiler=compiler, sample_size=20)
        approx_circuits: list[Circuit] = quest_runner.get_all_circuits(circuit=initial_circ)

    utries = [x.get_unitary() for x in approx_circuits]
    depths = [x.depth for x in approx_circuits]
    counts = [x.count(CNOTGate()) for x in approx_circuits]
    # dists = [x[1] for x in approx_circuits]

    # Store approximate solutions
    dir = f"ensemble_approx_circuits_frobenius/{method}/{circ_type}/{tol}/{timestep}"

    Path(dir).mkdir(parents=True, exist_ok=True)

    for i, circ in enumerate(approx_circuits):
        file = f"{dir}/circ_{i}.pickle"
        pickle.dump(circ, open(file, "wb"))

    summary = {}

    summary["orig_depth"] = orig_depth
    summary["orig_count"] = orig_count
    summary["depths"] = depths
    summary["counts"] = counts
    summary["avg_depth"] = np.mean(depths)
    summary["avg_count"] = np.mean(counts)

    json.dump(summary, open(f"{dir}/summary.json", "w"), indent=4)

    print(len(approx_circuits))
    # print(dists)
    # print(dists2)




    # for seed in range(1, 500):
        # out_circ = compile(target, optimization_level=3, error_threshold=err_thresh, seed=seed)

        # if out_circ not in synth_circs:
        #     synth_circs.append(out_circ)