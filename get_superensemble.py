from bqskit.ir.circuit import Circuit
from sys import argv
from bqskit.exec.runners.quest import QuestRunner
from bqskit.exec.runners.sim import SimulationRunner
from bqskit import compile
import numpy as np
from bqskit.compiler.compiler import Compiler
from bqskit.ir.point import CircuitPoint
from bqskit.ir.gates import CNOTGate, GlobalPhaseGate
# Generate a super ensemble for some error bounds
from bqskit.passes import *
from bqskit.runtime import get_runtime
import pickle
from bqskit.ir.opt.cost.functions import HilbertSchmidtResidualsGenerator, HilbertSchmidtCostGenerator, FrobeniusCostGenerator
from bqskit.ir.opt.minimizers.lbfgs import LBFGSMinimizer
from bqskit.ir.opt.minimizers.scipy import ScipyMinimizer
import multiprocessing as mp
from qfactorjax.qfactor import QFactorJax
from bqskit.ext import qiskit_to_bqskit, bqskit_to_qiskit
from bqskit.utils.math import global_phase, canonical_unitary, correction_factor


from bqskit import enable_logging

from pathlib import Path

import json

from os.path import join

def write_circ(circ_info):
    global basic_circ
    global target

    circuit, circ_dir, timestep,  circ_file, tol = circ_info

    full_dir = join(circ_dir, f"{tol}", f"{timestep}")

    Path(full_dir).mkdir(parents=True, exist_ok=True)

    full_file = join(full_dir, circ_file)

    pickle.dump(circuit, open(full_file, "wb"))
    return


def get_dist(circuit):
    global basic_circ
    dist = target.get_frobenius_distance(basic_circ.get_unitary(circuit))
    return dist

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
from jax import config
from bqskit.runtime import default_server_port
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import (NoiseModel, QuantumError, ReadoutError,
    pauli_error, depolarizing_error, thermal_relaxation_error)
enable_logging(False)

import logging
_logger = logging.getLogger(__name__)

def create_pauli_noise_model(rb_fid_1q, rb_fid_2q):
    rb_err_1q = 1 - rb_fid_1q
    rb_err_2q = 1 - rb_fid_2q

    # Create an empty noise model
    noise_model = NoiseModel()

    # Add depolarizing error to all single qubit u1, u2, u3 gates
    one_q_error = pauli_error([('X', rb_err_1q /3), ('Y', rb_err_1q /3), ('Z', rb_err_1q / 3), ('I', rb_fid_1q)])
    two_q_error = pauli_error([('XX', rb_err_2q / 15), ('YX', rb_err_2q / 15), ('ZX', rb_err_2q / 15),
                            ('XY', rb_err_2q / 15), ('YY', rb_err_2q / 15), ('ZY', rb_err_2q / 15),
                                ('XZ', rb_err_2q / 15), ('YZ', rb_err_2q / 15), ('ZZ', rb_err_2q / 15),
                                ('XI', rb_err_2q / 15), ('YI', rb_err_2q / 15), ('ZI', rb_err_2q / 15),
                                ('IX', rb_err_2q / 15), ('IY', rb_err_2q / 15), ('IZ', rb_err_2q / 15), 
                                ('II', rb_fid_2q)])
    noise_model.add_all_qubit_quantum_error(one_q_error, ['u3'])
    noise_model.add_all_qubit_quantum_error(two_q_error, ['cx'])

    return noise_model


# Circ 
if __name__ == '__main__':
    # enable_logging(True)
    global basic_circ
    global target

    final_tol = None
    np.set_printoptions(precision=4, threshold=np.inf, linewidth=np.inf)
    circ_type = argv[1]
    timestep = int(argv[2])
    method = argv[3]
    tol = int(argv[4])
    block_size = int(argv[5])
    if len(argv) == 8:
        prev_tol = argv[6]
        prev_block_size = argv[7]
    else:
        prev_tol = None
        prev_block_size = None


    detached_server_ip = 'localhost'
    detached_server_port = default_server_port
    config.update('jax_enable_x64', True)
    actual_target = None

    if circ_type == "TFIM":
        initial_circ = Circuit.from_file(f"/pscratch/sd/j/jkalloor/bqskit/ensemble_benchmarks/TFIM_3_timesteps/TFIM_3_{timestep}.qasm")
        initial_circ.remove_all_measurements()
        # target = np.loadtxt("/pscratch/sd/j/jkalloor/bqskit/ensemble_benchmarks/tfim_4-1.unitary", dtype=np.complex128)
    elif circ_type == "Heisenberg":
        initial_circ = Circuit.from_file("/pscratch/sd/j/jkalloor/bqskit/ensemble_benchmarks/heisenberg_3.qasm")
    elif circ_type == "Heisenberg_7" or circ_type == "TFXY_8":
        initial_circ = Circuit.from_file(f"/pscratch/sd/j/jkalloor/bqskit/ensemble_benchmarks/{circ_type}/{circ_type}_{timestep}.qasm")
        initial_circ.remove_all_measurements()
        if prev_tol:
            actual_initial_circ: Circuit = pickle.load(open(f"/pscratch/sd/j/jkalloor/bqskit/ensemble_approx_circuits_qfactor/gpu_real/{circ_type}/jiggled_circs/{prev_tol}/{prev_block_size}/{timestep}/jiggled_circ.pickle", "rb"))
            target = initial_circ.get_unitary()
            print("ORIG GPU DIST", actual_initial_circ.get_unitary().get_frobenius_distance(target))
            print(f"Orig Depth: {initial_circ.depth}, New Depth: {actual_initial_circ.depth}")
            print(f"Orig Count: {initial_circ.num_operations}, New Count: {actual_initial_circ.num_operations}")
            initial_circ = actual_initial_circ

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


    synth_circs = []

    # TODO: Divide by number of blocks yo
    err_thresh = 10 ** (-1 * tol)
    extra_err_thresh = 1e-13

    orig_depth = initial_circ.depth
    orig_count = initial_circ.count(CNOTGate())

    # workflow = [
    #     QFASTDecompositionPass(),
    #     ForEachBlockPass([LEAPSynthesisPass(), ScanningGateRemovalPass()]),
    #     UnfoldPass(),
    # ]

    # compiler = Compiler(detached_server_ip, detached_server_port, runtime_log_level=logging.DEBUG)
    # if block_size == 5:
    #     num_workers = 10
    # elif block_size == 6:
    #     num_workers = 4
    # elif block_size == 7:
    #     num_workers = 2
    # else:
    #     num_workers = 20
    num_workers = 4
    compiler = Compiler(num_workers=num_workers)
    
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
        dist_tol = err_thresh * 10e-5

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

        gate_deletion_jax_pass = ScanningGateRemovalPass( instantiate_options=instantiate_options)
        
        workflow = [
            # ToU3Pass(convert_all_single_qubit_gates=True),
            ToVariablePass(convert_all_single_qudit_gates=True),
            QuickPartitioner(block_size=block_size),
            ForEachBlockPass([
                ScanningGateRemovalPass(
                    instantiate_options=instantiate_options,
                ),
            ]),
            UnfoldPass(),
            ToU3Pass(),
        ]
        out_circ, data = compiler.compile(initial_circ, workflow, request_data=True)
        new_target = out_circ.get_unitary().get_target_correction_factor(target) * target
        out_canonical = UnitaryMatrix(canonical_unitary(out_circ.get_unitary()))

        print("FINAL GPU Dist: ", out_circ.get_unitary().get_frobenius_distance(new_target))
        print("FINAL GPU Dist: ", out_circ.get_unitary().get_distance_from(target, degree=1)*(2 ** 8))
        dir = f"ensemble_approx_circuits_qfactor/{method}_tighter/{circ_type}"
        Path(f"{dir}/jiggled_circs/{tol}/{block_size}/{timestep}").mkdir(parents=True, exist_ok=True)
        pickle.dump(out_circ, open(f"{dir}/jiggled_circs/{tol}/{block_size}/{timestep}/jiggled_circ.pickle", "wb"))
        pickle.dump(new_target, open(f"{dir}/jiggled_circs/{tol}/{block_size}/{timestep}/target.pickle", "wb"))
        exit(0)

    elif method == "jiggle":


        workflow = [
            # ToU3Pass(convert_all_single_qubit_gates=True),
            # ScanPartitioner(3),
            # ForEachBlockPass([
            #     synthesis_pass
            # ],
            # replace_filter="less-than"),
            # UnfoldPass(),
            JiggleEnsemblePass(success_threshold=err_thresh, num_circs=10000, cost=generator)
        ]
        out_circ, data = compiler.compile(initial_circ, workflow, request_data=True)
        print(out_circ.num_cycles)
        approx_circuits: list[Circuit] = data["ensemble_params"]
        # approx_circuits = data["ensemble"]


    elif method == "noise":
        dir = f"ensemble_approx_circuits_qfactor/{method}/{circ_type}/{tol}/{block_size}"
        Path(f"{dir}").mkdir(parents=True, exist_ok=True)
        one_q_prob = 0.001
        two_q_prob = 0.001
        # for i in range(20000):
        #     # Calculate total unitary
        #     # utry = initial_circ.get_noisy_pauli_unitary(0.001, 0)
        #     pickle.dump(utry, open(f"{dir}/utry_{i}.pickle", "wb"))

        def save_noisy_unitary(i: int):
            utry = initial_circ.get_noisy_pauli_unitary(0.1, 0)
            # print(utry)
            pickle.dump(utry, open(f"{dir}/utry_{i}.pickle", "wb"))

        with mp.Pool() as pool:
            utries = pool.map(save_noisy_unitary, range(2000))

        exit(0)

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
    dir = f"ensemble_approx_circuits_qfactor/{method}_post_gpu/{circ_type}"

    Path(f"{dir}/jiggled_circs/{tol}/{block_size}/{timestep}").mkdir(parents=True, exist_ok=True)

    # Get first 100 dists
    basic_circ = out_circ
    with mp.Pool() as pool:
        dists = pool.map(get_dist, approx_circuits[:500])


    actual_tol = int(-1 * np.log10(np.mean(dists)))
    print(actual_tol)

    circ_infos = [(circ, dir, timestep, f"params_{i}_{block_size}_{tol}.pickle", actual_tol) for i, circ in enumerate(approx_circuits)]
    print("Writing")
    with mp.Pool() as pool:
        pool.map(write_circ, circ_infos)


    # if method.startswith("jiggle"):
    #     pickle.dump(out_circ, open(f"{dir}/jiggled_circs/{tol}/{block_size}/{timestep}/jiggled_circ.pickle", "wb"))

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

    # print(len(approx_circuits))
    # print(dists)
    # print(dists2)




    # for seed in range(1, 500):
        # out_circ = compile(target, optimization_level=3, error_threshold=err_thresh, seed=seed)

        # if out_circ not in synth_circs:
        #     synth_circs.append(out_circ)