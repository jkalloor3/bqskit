from bqskit.ir import Circuit
from bqskit.ir.gates import CNOTGate, GlobalPhaseGate, VariableUnitaryGate
from bqskit.passes import ScanPartitioner, ForEachBlockPass, CheckpointRestartPass, SimpleLayerGenerator, NOOPPass, ToVariablePass
from util import HamiltonianNoisePass, EnsembleLeap, frobenius_cost, EnsembleZXZXZ
from bqskit.compiler import Compiler
import numpy as np
from sys import argv
from bqskit import enable_logging

enable_logging(True)


def create_worflow(qasm_name: str, 
                   noise_epsilon: float, 
                   synthesis_epsilon: float, 
                   ensemble_size: int,
                   zxzxz: bool = False) -> list:

    if zxzxz:
        num_noisy = ensemble_size
    else:
        num_noisy = ensemble_size
        
    perturbation_pass = HamiltonianNoisePass(num_noisy, noise_epsilon, kmeans=False)  # Perturb the Hamiltonians

    if not zxzxz:
        checkpoint_dir = f"hamiltonian_perturbation_checkpoints_leap/{qasm_name}_{noise_epsilon}_{synthesis_epsilon}_{ensemble_size}/"
        checkpoint_pass = CheckpointRestartPass(checkpoint_dir, 
                                                default_passes=[ScanPartitioner(3)])
        
        instantiation_options = {
            "method": "qfactor",
        }

        synthesis_pass = EnsembleLeap(
            success_threshold=synthesis_epsilon,
            synthesize_perturbations_only=True,
            # use_calculated_error=True,
            partial_success_threshold=noise_epsilon,
            max_psols=1,
            store_partial_solutions=True,
            max_layer=14,
            # layer_generator=layer_gen,
            instantiate_options=instantiation_options
        )
    else:
        checkpoint_dir = f"hamiltonian_perturbation_checkpoints_zxzxz/{qasm_name}_{noise_epsilon}_{synthesis_epsilon}_{ensemble_size}/"
        checkpoint_pass = CheckpointRestartPass(checkpoint_dir, 
                                                default_passes=[NOOPPass()])
        tree_depth = max(int(np.log2(256 // ensemble_size)), 1)
        synthesis_pass = EnsembleZXZXZ(
            extract_diagonal=True,
            synthesis_epsilon=synthesis_epsilon,
            tree_depth=tree_depth
        )

    # Create the workflow
    inner_workflow = [
        perturbation_pass, 
        ToVariablePass(),
        synthesis_pass
    ]

    if zxzxz:
        workflow = [checkpoint_pass] + inner_workflow
    else:
        # Define the workflow passes
        workflow = [
            checkpoint_pass,
            ForEachBlockPass(inner_workflow,
                            allocate_error=True,
                            allocate_error_gate=CNOTGate())
        ]

    return workflow

if __name__ == '__main__':
    # Load circuit from qasm file
    qasm_name = "adder9_hard_qfactor"
    # qasm_file = '/pscratch/sd/j/jkalloor/bqskit/fixed_block_checkpoints_min/adder9_0_2_8_3/block_2.qasm'
    # qasm_file = "/pscratch/sd/j/jkalloor/bqskit/ensemble_benchmarks/tfxy_6.qasm"
    qasm_file = "/pscratch/sd/j/jkalloor/bqskit/checkpoint_qasms/adder9_8_3/block_2.qasm"
    circuit = Circuit.from_file(qasm_file)
    noise_log = int(argv[1])
    zxzxz = bool(int(argv[2]))
    max_synth = int(argv[3]) if len(argv) > 3 else 1

    # Range syntheis epsilon from noise_epsilon / 10 to noise_epsilon ** 2 + 1
    noise_epsilon = 10 ** (noise_log * -1)
    synth_epsilons = [10 ** (noise_log * -1 - i) for i in range(1, max_synth + 1)]
    # synth_epsilons = [noise_epsilon / (10 ** 2)]1
    compiler = Compiler(num_workers=256)
    for synthesis_epsilon in synth_epsilons:
        for ensemble_size in [32]:
        # for ensemble_size in [8 * i for i  in range(3, 10, 2)]:
            # Create the workflow
            workflow = create_worflow(qasm_name,
                                    noise_epsilon, 
                                    synthesis_epsilon, 
                                    ensemble_size,
                                    zxzxz=zxzxz)

            compiler.compile(circuit, workflow=workflow)