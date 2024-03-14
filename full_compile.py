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
import multiprocessing as mp
from bqskit.ext import qiskit_to_bqskit

from bqskit import enable_logging, compile

from pathlib import Path

import json

from os.path import join

enable_logging(True)

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
    np.set_printoptions(precision=4, threshold=np.inf, linewidth=np.inf)
    circ_type = argv[1]
    
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
    elif circ_type == "TFXY":
        initial_circ = Circuit.from_file("/pscratch/sd/j/jkalloor/bqskit/ensemble_benchmarks/tfxy_6.qasm")
    elif circ_type == "TFXY_t":
        initial_circ: Circuit =  Circuit.from_file(f"/pscratch/sd/j/jkalloor/bqskit/ensemble_benchmarks/TFXY_5_timesteps/TFXY_5_{timestep}.qasm")
        initial_circ.remove_all_measurements()
    else:
        target = np.loadtxt("/pscratch/sd/j/jkalloor/bqskit/ensemble_benchmarks/qite_3.unitary", dtype=np.complex128)
        initial_circ = Circuit.from_unitary(target)

    target = initial_circ.get_unitary()
    # print(initial_circ)

    synth_circs = []

    orig_depth = initial_circ.depth
    orig_count = initial_circ.count(CNOTGate())

    compiler = Compiler(num_workers=-1)

    generator = HilbertSchmidtCostGenerator()
    synthesis_pass = LEAPSynthesisPass(
        success_threshold = 1e-14,
        cost=generator,
        instantiate_options={
            'min_iters': 100,
            # 'ftol': 1e-15,
            # 'dist_tol': 1e-15,
            # 'gtol': 1e-10,
            # 'cost_fn_gen': generator,
            # 'method': 'qfactor',
            'method': 'minimization',
            'minimizer': LBFGSMinimizer() # Go back to QFactor. set x_tol
        }
    )

    workflow = [
        # ToU3Pass(convert_all_single_qubit_gates=True),
        ScanPartitioner(3),
        ForEachBlockPass([
            synthesis_pass
        ],
        replace_filter="less-than"),
        UnfoldPass(),
    ]
    out_circ = compiler.compile(initial_circ, workflow)

    dist = target.get_frobenius_distance(out_circ.get_unitary())

    print("Original Depth: ", orig_depth)
    print("Original Count: ", orig_count)
    print("New Depth: ", out_circ.depth)
    print("New Count: ", out_circ.count(CNOTGate()))
    print("Dist: ", dist)


    full_dir = f"ensemble_benchmarks_shorter/"

    Path(full_dir).mkdir(parents=True, exist_ok=True)

    full_file = join(full_dir, f"{circ_type}.pkl")

    pickle.dump(out_circ, open(full_file, "wb"))
