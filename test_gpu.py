from bqskit import Circuit
from qfactorjax.qfactor_sample_jax import QFactorSampleJax
from bqskit.passes import *
from bqskit.compiler import Compiler
from bqskit import enable_logging

enable_logging(True)

# Load a circuit from QASM
in_circuit = Circuit.from_file("ensemble_benchmarks/qc_optimized_5q.qasm")

num_multistarts = 16

qfactor_sample_gpu_instantiator = QFactorSampleJax()

instantiate_options = {
        'method': qfactor_sample_gpu_instantiator,
        'multistarts': num_multistarts,
    }

# Prepare the compilation passes
passes = [
    # Convert U3s to VU
    ToVariablePass(convert_all_single_qudit_gates=True),

    ScanningGateRemovalPass(
        # tree_depth=2,
        instantiate_options=instantiate_options,
        # store_all_solutions=True,
    ),

    # Combine the partitions back into a circuit
    UnfoldPass(),

    # Convert back the VariablueUnitaires into U3s
    ToU3Pass(),
]



with Compiler(num_workers=1) as compiler:
    out_circuit = compiler.compile(in_circuit, passes)
    print(
            f'Circuit finished with gates: {out_circuit.gate_counts}, '
            f'while started with {in_circuit.gate_counts}',
        )