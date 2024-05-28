from bqskit.ir.circuit import Circuit
from sys import argv
import numpy as np
from bqskit.compiler.compiler import Compiler
from bqskit.ir.gates import CNOTGate
# Generate a super ensemble for some error bounds
from bqskit.passes import *
from bqskit.runtime import get_runtime
from bqskit.ir.gates import GlobalPhaseGate
import pickle
from bqskit.ir.opt.cost.functions import HilbertSchmidtResidualsGenerator, HilbertSchmidtCostGenerator
from qfactorjax.qfactor import QFactorJax

from bqskit import enable_logging

from pathlib import Path

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
    if len(argv) == 8:
        prev_tol = argv[6]
        prev_block_size = argv[7]
    else:
        prev_tol = None
        prev_block_size = None

    detached_server_ip = 'localhost'
    detached_server_port = default_server_port
    config.update('jax_enable_x64', True)



    if circ_type.startswith("vqe"):
        initial_circ = Circuit.from_file(f"/pscratch/sd/j/jkalloor/bqskit/ensemble_benchmarks/{circ_type}.qasm")
    else:
        initial_circ = Circuit.from_file(f"/pscratch/sd/j/jkalloor/bqskit/ensemble_benchmarks/{circ_type}/{circ_type}_{timestep}.qasm")
        initial_circ.remove_all_measurements()
    
    if initial_circ.num_qudits < 8:
        target = initial_circ.get_unitary()

    pam_circ: Circuit = pickle.load(open(f"/pscratch/sd/j/jkalloor/bqskit/ensemble_approx_circuits_pam/pam/{circ_type}/{prev_tol}/{prev_block_size}/{timestep}/circ.pickle", "rb"))

    orig_depth = initial_circ.depth
    orig_count = initial_circ.count(CNOTGate())
                                    
    pam_depth = pam_circ.depth
    pam_count = pam_circ.count(CNOTGate())

    print("COUNT", "DEPTH")
    print(orig_count, orig_depth)
    print("PAM COUNT", "PAM DEPTH")
    print(pam_count, pam_depth)

    initial_circ = pam_circ
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

    if method == "gpu":
        num_multistarts = 32
        max_iters = 100000
        min_iters = 3
        diff_tol_r = 1e-5
        diff_tol_a = 0.0
        approx_num_blocks = max((initial_circ.num_qudits / block_size) * initial_circ.depth / 20, 1)
        dist_tol = err_thresh #/ approx_num_blocks
        # print(dist_tol)

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

        base_checkpoint_dir = "checkpoints"
        proj_name = f"{circ_type}_{method}_{timestep}_{block_size}_{tol}"
        full_checkpoint_dir = join(base_checkpoint_dir, proj_name)

        if block_size == initial_circ.num_qudits:
            # Only 1 block, do TreeScan
            gate_deletion_jax_pass = TreeScanningGateRemovalPass(instantiate_options=instantiate_options, tree_depth=8)
        else:
            gate_deletion_jax_pass = ScanningGateRemovalPass(instantiate_options=instantiate_options, checkpoint_proj=full_checkpoint_dir)
        print(block_size)
        # Check if checkpoint files exist
        workflow = [
            CheckpointRestartPass(base_checkpoint_dir, proj_name, 
                                  default_passes=[
                                    ToVariablePass(convert_all_single_qudit_gates=True),
                                    QuickPartitioner(block_size=block_size),
                                  ], save_as_qasm=False),
            ForEachBlockPass([
                gate_deletion_jax_pass,
            ]),
            UnfoldPass(),
            ToU3Pass(),
        ]
        
        out_circ, data = compiler.compile(initial_circ, workflow, request_data=True)
        if out_circ.num_qudits < 8:
            global_phase_correction = target.get_target_correction_factor(out_circ.get_unitary())
            out_circ.append_gate(GlobalPhaseGate(1, global_phase=global_phase_correction), (0,))
            print("FINAL Original GPU Dist: ", out_circ.get_unitary().get_frobenius_distance(target))
        dir = f"ensemble_approx_circuits_qfactor/{method}_post_pam/{circ_type}"
        Path(f"{dir}/jiggled_circs/{tol}/{block_size}/{timestep}").mkdir(parents=True, exist_ok=True)
        pickle.dump(out_circ, open(f"{dir}/jiggled_circs/{tol}/{block_size}/{timestep}/jiggled_circ.pickle", "wb"))
