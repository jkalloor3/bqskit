from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit
from sys import argv
from bqskit.exec.runners.quest import QuestRunner
from bqskit.exec.runners.sim import SimulationRunner
from bqskit import compile
import numpy as np
from bqskit.compiler.compiler import Compiler, WorkflowLike
from bqskit.ir.point import CircuitPoint
from bqskit.ir.gates import CNOTGate, GlobalPhaseGate, VariableUnitaryGate
# Generate a super ensemble for some error bounds
from bqskit.passes import *
from bqskit.runtime import get_runtime
import pickle
from bqskit.ir.opt.cost.functions import  HilbertSchmidtCostGenerator, FrobeniusNoPhaseCostGenerator
from bqskit.passes import ScanningGateRemovalPass, IfThenElsePass, PassPredicate


from util import SecondLEAPSynthesisPass, GenerateProbabilityPass, SelectFinalEnsemblePass, LEAPSynthesisPass2, SubselectEnsemblePass
from util import WriteQasmPass, ReplaceWithQasmPass, CheckEnsembleQualityPass, HamiltonianNoisePass, EnsembleLeap, EnsembleZXZXZ

from bqskit import enable_logging

from util import save_circuits, load_circuit, FixGlobalPhasePass, CalculateErrorBoundPass


class BadEnsemblePredicate(PassPredicate):

    def __init__(self, remove_ensemble: bool = True) -> None:
        self.remove_ensemble = remove_ensemble

    def get_truth_value(self, circuit: Circuit, data: PassData) -> bool:
        has_good_ensemble = data.get("good_ensemble", False)
        if not has_good_ensemble:
            print("NO GOOD ENSEMBLES, TRYING AGAIN!", flush=True)
            data.pop("good_ensemble", None)
            block_data = data[ForEachBlockPass.key]
            previous_data_key = ForEachBlockPass.key + "_previous"
            if previous_data_key in data:
                data[previous_data_key].append(block_data[0])
            else:
                data[previous_data_key] = [block_data[0]]
            data[ForEachBlockPass.key] = []
            if self.remove_ensemble:
                data.pop("ensemble", None)
                data.pop("scan_sols", None)
        else:
            print("GOOD ENSEMBLES FOUND!", flush=True)
        return not has_good_ensemble

def get_distance(circ1: Circuit) -> float:
    global target
    return circ1.get_unitary().get_frobenius_distance(target)


def get_shortest_circuits(circ_name: str, tol: int, timestep: int,
                          num_unique_circs: int = 100, extra_str="") -> list[Circuit]:
    circ = load_circuit(circ_name, opt=opt)

    print("Original CNOT Count: ", circ.count(CNOTGate()))
    
    # workflow = gpu_workflow(tol, f"{circ_name}_{tol}_{timestep}")
    if tol == 0:
        err_thresh = 0.2
    else:
        err_thresh = 10 ** (-1 * tol)
        
    extra_err_thresh = 1e-2 * err_thresh
    big_block_size = 8
    small_block_size = 3
    checkpoint_dir = f"fixed_block_checkpoints_min{extra_str}/{circ_name}_{timestep}_{tol}_{big_block_size}_{small_block_size}/"

    fast_instantiation_options = {
        'multistarts': 2,
        'ftol': extra_err_thresh,
        'diff_tol_r': 1e-4,
        'max_iters': 10000,
        'min_iters': 100,
        # 'method': 'minimization',
        # 'method': 'qfactor',
    }

    good_instantiation_options = {
        'multistarts': 4,
        'ftol': 5e-16,
        'gtol': 1e-15,
        'diff_tol_r': 1e-6,
        'max_iters': 100000,
        'min_iters': 1000,
    }

    slow_partitioner_passes = [
        ScanPartitioner(block_size=small_block_size),
        ExtendBlockSizePass(),
        ScanPartitioner(block_size=big_block_size),
        ExtendBlockSizePass(),
    ]

    fast_partitioner_passes = [
        QuickPartitioner(block_size=small_block_size),
        ExtendBlockSizePass(),
        QuickPartitioner(block_size=big_block_size),
        ExtendBlockSizePass(),
    ]

    if circ.num_qudits > 20:
        partitioner_passes = fast_partitioner_passes
        instantiation_options = fast_instantiation_options
    else:
        partitioner_passes = slow_partitioner_passes
        instantiation_options = good_instantiation_options


    synthesis_pass = LEAPSynthesisPass2(
        store_partial_solutions=True,
        # layer_generator=layer_gen,
        success_threshold = extra_err_thresh,
        partial_success_threshold=err_thresh,
        # cost=phase_generator,
        instantiate_options=instantiation_options,
        use_calculated_error=True,
        max_layer=12,
        max_psols=6
    )

    second_synthesis_pass = SecondLEAPSynthesisPass(
        success_threshold = extra_err_thresh,
        # layer_generator=layer_gen,
        partial_success_threshold=err_thresh,
        # cost=HS,
        instantiate_options=instantiation_options,
        use_calculated_error=True,
        max_layer=14,
        max_psols=4
    )

    def create_ensemble_pass(z: str):
        return CreateEnsemblePass(
            success_threshold=err_thresh, 
            use_calculated_error=True, 
            num_circs=num_unique_circs,
            num_random_ensembles=6,
            solve_exact_dists=True,
            checkpoint_extra_str=z
        )
    
    def create_jiggle_pass(z: str):
        return JiggleEnsemblePass(success_threshold=err_thresh, 
                                  num_circs=5000, 
                                  use_ensemble=True,
                                  use_calculated_error=True,
                                  checkpoint_extra_str=z)

    # INITIAL TRY TO GENERATE ENSEMBLE
    best_ensemble_generation_workflow = [
        ForEachBlockPass(
            [
                synthesis_pass,
                second_synthesis_pass,
                FixGlobalPhasePass(),
            ],
            calculate_error_bound=False,
            allocate_error=True,
            allocate_error_gate=CNOTGate(),
            allocate_skew_factor=0.5, # Skew for harder blocks
        ),
        create_ensemble_pass("_best"),
        create_jiggle_pass("_best"),
        CheckEnsembleQualityPass(count_t=False, csv_name="_best"),
    ]

    # SECOND TRY TO GENERATE ENSEMBLE
    perturbation_pass = HamiltonianNoisePass(num_unique_circs, err_thresh, use_calculated_error=True)
    ens_synthesis_pass = EnsembleLeap(success_threshold=extra_err_thresh, 
                                  use_calculated_error=True,
                                  partial_success_threshold=err_thresh,
                                  max_psols=1,
                                  store_partial_solutions=True,
                                  max_layer=14)
    
    ens_synthesis_pass2 = EnsembleLeap(success_threshold=extra_err_thresh, 
                                use_calculated_error=True,
                                partial_success_threshold=err_thresh,
                                synthesize_perturbations_only=True,
                                max_psols=1,
                                store_partial_solutions=True,
                                max_layer=14)
    
    second_ensemble_generation_workflow = [
        ForEachBlockPass(
            [
                perturbation_pass,
                ens_synthesis_pass,
                # FixGlobalPhasePass(),
            ],
            calculate_error_bound=True,
            allocate_error=True,
            allocate_error_gate=CNOTGate(),
        ),
        create_ensemble_pass("_second"),
        # create_jiggle_pass("_second"),
        CheckEnsembleQualityPass(count_t=False, csv_name="_second"),
    ]

    # THIRD TRY TO GENERATE ENSEMBLE
    third_ensemble_generation_workflow = [
        ForEachBlockPass(
            [
                perturbation_pass,
                ens_synthesis_pass2,
                # FixGlobalPhasePass(),
            ],
            calculate_error_bound=True,
            allocate_error=True,
            allocate_error_gate=CNOTGate(),
        ),
        create_ensemble_pass("_third"),
        # create_jiggle_pass("_second"),
        CheckEnsembleQualityPass(count_t=False, csv_name="_second"),
    ] 

    # LAST TRY TO GENERATE ENSEMBLE
    zxzxz_pass = EnsembleZXZXZ(
            extract_diagonal=True,
            synthesis_epsilon=extra_err_thresh,
            tree_depth=4
    )

    # Runs on the large block
    last_ensemble_generation_workflow = [
        perturbation_pass,
        zxzxz_pass,
        # FixGlobalPhasePass(),
        create_jiggle_pass("_last"),
        CheckEnsembleQualityPass(count_t=False, csv_name="_last"),
    ]

    leap_workflow = [
        CheckpointRestartPass(checkpoint_dir, 
                                default_passes=partitioner_passes),
        ForEachBlockPass(
            [
                WriteQasmPass(),
                # *best_ensemble_generation_workflow,
                # IfThenElsePass(BadEnsemblePredicate(remove_ensemble=True), 
                #     second_ensemble_generation_workflow,
                #     [NOOPPass()],
                # ),
                # IfThenElsePass(BadEnsemblePredicate(remove_ensemble=True), 
                #     third_ensemble_generation_workflow,
                #     [NOOPPass()],
                # ),
                # SubselectEnsemblePass(success_threshold=err_thresh, num_circs=200),
                # GenerateProbabilityPass(success_threshold=err_thresh, size=50),
            ],
            calculate_error_bound=True,
            allocate_error=True,
            allocate_error_gate=CNOTGate(),
            allocate_skew_factor=1 # Skew for bigger blocks since they have more sub-blocks
        ),
        # SelectFinalEnsemblePass(size=5000),
    ]
    num_workers = 20
    compiler = Compiler(num_workers=num_workers)
    # target = circ.get_unitary()
    out_circ, data = compiler.compile(circ, workflow=leap_workflow, request_data=True)
    approx_circuits: list[Circuit] = data["final_ensemble"]
    print("Num Circs", len(approx_circuits))
    return approx_circuits, data.error, out_circ.count(CNOTGate())

if __name__ == '__main__':
    global target
    circ_name = argv[1]
    timestep = int(argv[2])
    tol = int(argv[3])
    num_unique_circs = int(argv[4])
    opt = bool(int(argv[5])) if len(argv) > 5 else False
    opt_str = "_opt" if opt else ""
    # print("OPT STR", opt_str, opt, argv[5])
    circs, error_bound, count = get_shortest_circuits(circ_name, tol, timestep, num_unique_circs=num_unique_circs, extra_str=f"{opt_str}")
    save_circuits(circs, circ_name, tol, timestep, ignore_timestep=True, extra_str=f"_{num_unique_circs}_circ_final_min_post{opt_str}_calc_bias")