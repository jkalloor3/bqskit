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
from bqskit.ir.gates import GlobalPhaseGate
import pickle
from bqskit.ir.opt.cost.functions import HilbertSchmidtResidualsGenerator, HilbertSchmidtCostGenerator, FrobeniusCostGenerator
from bqskit.ir.opt.minimizers.lbfgs import LBFGSMinimizer
from bqskit.ir.opt.minimizers.scipy import ScipyMinimizer
import multiprocessing as mp
from qfactorjax.qfactor import QFactorJax
from bqskit.ext import qiskit_to_bqskit
from bqskit.utils.math import global_phase, canonical_unitary, correction_factor

from bqskit import enable_logging

from pathlib import Path

import json

from os.path import join

def write_circ(circ_info):
    global basic_circ
    global target

    circuit, circ_dir, timestep,  circ_file = circ_info
    dist = target.get_frobenius_distance(basic_circ.get_unitary(circuit))
    if dist < 1e-18:
        return

    tol = int(-1 * np.log10(dist))
    print(tol)

    full_dir = join(circ_dir, f"{tol}", f"{timestep}")

    Path(full_dir).mkdir(parents=True, exist_ok=True)

    full_file = join(full_dir, circ_file)

    pickle.dump(circuit, open(full_file, "wb"))
    return

from bqskit.ir.opt.cost.differentiable import DifferentiableCostFunction
from bqskit.ir.opt.cost.generator import CostFunctionGenerator
from bqskit.qis.state.state import StateVector
from bqskit.qis.state.system import StateSystem

from bqskit.ir.circuit import Circuit
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.ir.opt.cost.function import CostFunction
from bqskit.qis.unitary.unitary import RealVector
import numpy.typing as npt
from jax import config
from bqskit.runtime import default_server_port

enable_logging(False)

import logging
_logger = logging.getLogger(__name__)

# Circ 
if __name__ == '__main__':
    # enable_logging(True)
    global basic_circ
    global target
    np.set_printoptions(precision=4, threshold=np.inf, linewidth=np.inf)
    circ_type = argv[1]
    timestep = int(argv[2])
    method = argv[3]
    tol = int(argv[4])
    block_size = int(argv[5])

    print(tol, block_size)

    detached_server_ip = 'localhost'
    detached_server_port = default_server_port
    config.update('jax_enable_x64', True)

    # q_circ = pickle.load(open(f"/pscratch/sd/j/jkalloor/bqskit/ensemble_benchmarks/{circ_type}/{circ_type}_{timestep}.pkl", "rb"))
    # initial_circ = qiskit_to_bqskit(q_circ)
    initial_circ = Circuit.from_file(f"/pscratch/sd/j/jkalloor/bqskit/ensemble_benchmarks/{circ_type}/{circ_type}_{timestep}.qasm")
    initial_circ.remove_all_measurements()
    target = initial_circ.get_unitary()

    orig_depth = initial_circ.depth
    orig_count = initial_circ.count(CNOTGate())

    print(orig_count, orig_depth)
    # print(initial_circ)

    synth_circs = []

    # TODO: Divide by number of blocks yo
    err_thresh = 10 ** (-1 * tol)
    extra_err_thresh = 1e-13

    if block_size == 5:
        num_workers = 10
    elif block_size == 6:
        num_workers = 4
    elif block_size == 7:
        num_workers = 2
    elif block_size == 8:
        num_workers = 1
    else:
        num_workers = -1

    compiler = Compiler(ip=detached_server_ip, port=detached_server_port)

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
            CreateEnsemblePass(success_threshold=err_thresh, num_circs=20000, cost=generator)
        ]

        old_workflow = [synthesis_pass, CreateEnsemblePass(success_threshold=err_thresh, num_circs=10000)]
        
        out_circ, data = compiler.compile(initial_circ, workflow, request_data=True)
        approx_circuits: list[Circuit] = data["ensemble"]
    elif method == "treescan":
        workflow = [
            ToVariablePass(convert_all_single_qudit_gates=True),
            ScanPartitioner(3),
            ForEachBlockPass([
                TreeScanningGateRemovalPass(success_threshold=err_thresh, store_all_solutions=True, tree_depth=7, cost=generator),
            ]),
            CreateEnsemblePass(success_threshold=err_thresh, num_circs=20000, cost=generator)
        ]


        out_circ, data = compiler.compile(initial_circ, workflow=workflow, request_data=True)
        approx_circuits: list[Circuit] = data["ensemble"]
    elif method == "quest":
        quest_runner = QuestRunner(SimulationRunner(), compiler=compiler, sample_size=20)
        approx_circuits: list[Circuit] = quest_runner.get_all_circuits(circuit=initial_circ)
    elif method == "gpu":
        num_multistarts = 32
        max_iters = 100000
        min_iters = 3
        diff_tol_r = 1e-5
        diff_tol_a = 0.0
        approx_num_blocks = max((initial_circ.num_qudits / block_size) * initial_circ.depth / 20, 1)
        dist_tol = err_thresh / approx_num_blocks
        print(dist_tol)

        diff_tol_step_r = 0.1
        diff_tol_step = 200
        beta = 0

        batched_instantiation = QFactorJax(
            diff_tol_r=diff_tol_r,
            diff_tol_a=diff_tol_a,
            min_iters=min_iters,
            max_iters=max_iters,
            dist_tol=dist_tol,
            diff_tol_step_r=diff_tol_step_r,
            diff_tol_step=diff_tol_step,
            beta=beta,
        )
        instantiate_options = {
            'method': batched_instantiation,
            'multistarts': num_multistarts,
        }
        if block_size == initial_circ.num_qudits:
            # Only 1 block, do TreeScan
            gate_deletion_jax_pass = TreeScanningGateRemovalPass(instantiate_options=instantiate_options, tree_depth=8)
        else:
            gate_deletion_jax_pass = ScanningGateRemovalPass(instantiate_options=instantiate_options)
        print(block_size)
        workflow = [
            ToVariablePass(convert_all_single_qudit_gates=True),
            QuickPartitioner(block_size=block_size),
            ForEachBlockPass([
                gate_deletion_jax_pass,
            ]),
            UnfoldPass(),
            ToU3Pass(),
        ]
        out_circ, data = compiler.compile(initial_circ, workflow, request_data=True)
        global_phase_correction = target.get_target_correction_factor(out_circ.get_unitary())

        out_circ.append_gate(GlobalPhaseGate(1, global_phase=global_phase_correction), (0,))

        print("FINAL Original GPU Dist: ", out_circ.get_unitary().get_frobenius_distance(target))
        dir = f"ensemble_approx_circuits_qfactor/{method}_real/{circ_type}"
        Path(f"{dir}/jiggled_circs/{tol}/{block_size}/{timestep}").mkdir(parents=True, exist_ok=True)
        pickle.dump(out_circ, open(f"{dir}/jiggled_circs/{tol}/{block_size}/{timestep}/jiggled_circ.pickle", "wb"))
        exit(0)
    elif method == "jiggle":
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
            JiggleEnsemblePass(success_threshold=err_thresh, num_circs=10000, cost=generator)
        ]
        out_circ, data = compiler.compile(initial_circ, workflow, request_data=True)
        print(out_circ.num_cycles)
        approx_circuits: list[Circuit] = data["ensemble_params"]
        # approx_circuits = data["ensemble"]

    # utries = [x.get_unitary() for x in approx_circuits]
    # dists = [get_frobenius_distance(x, target) for x in approx_circuits]
    # print(dists)
    # print("Getting Data")
    # with mp.Pool() as pool:
    #     depths = pool.map(lambda x: x.depth, approx_circuits)
    #     counts = pool.map(lambda x: x.count(CNOTGate()), approx_circuits) 

    # depths = [x.depth for x in approx_circuits]
    # counts = [x.count(CNOTGate()) for x in approx_circuits]
    # dists = [x[1] for x in approx_circuits]

    # Store approximate solutions
    dir = f"ensemble_approx_circuits_qfactor/{method}_real/{circ_type}"

    Path(f"{dir}/jiggled_circs/{tol}/{timestep}").mkdir(parents=True, exist_ok=True)

    circ_infos = [(circ, dir, timestep, f"params_{i}_{tol}.pickle") for i, circ in enumerate(approx_circuits)]
    print("Writing")
    basic_circ = out_circ
    with mp.Pool() as pool:
        pool.map(write_circ, circ_infos)



    if method == "jiggle":
        pickle.dump(out_circ, open(f"{dir}/jiggled_circs/{tol}/{timestep}/jiggled_circ.pickle", "wb"))

    # for i, circ in enumerate(approx_circuits):
    #     file = f"{dir}/circ_{i}.pickle"
    #     pickle.dump(circ, open(file, "wb"))

    # summary = {}

    # summary["orig_depth"] = orig_depth
    # summary["orig_count"] = orig_count
    # summary["depths"] = depths
    # summary["counts"] = counts
    # summary["avg_depth"] = np.mean(depths)
    # summary["avg_count"] = np.mean(counts)

    # json.dump(summary, open(f"{dir}/summary.json", "w"), indent=4)

    print(len(approx_circuits))
    # print(dists)
    # print(dists2)




    # for seed in range(1, 500):
        # out_circ = compile(target, optimization_level=3, error_threshold=err_thresh, seed=seed)

        # if out_circ not in synth_circs:
        #     synth_circs.append(out_circ)