import numpy as np
from bqskit.ir import Circuit
from bqskit.ir.gates import U3Gate, CNOTGate


from bqskit.passes import LEAPSynthesisPass, UpdateDataPass, ScanPartitioner, ForEachBlockPass, DiversifyEnsemblePass, GenerateProbabilitiesPass, BiasFilterPass, GenerateCircuitSamplerPass
from bqskit.compiler import Compiler

if __name__ == '__main__':
    qasm_name = "qae_7"
    input_circ = Circuit.from_file(qasm_name + '.qasm')
    compiler = Compiler(num_workers=255)

    # Create a partitioned circuit
    # When using bqskit `compile`, we use UpdateData to tell the compiler we
    # want to compile an ensemble of solutions 

    # We are using block size 4 for better results. This will lead to longer run
    # times
    partitioned_circ, data = compiler.compile(input_circ, 
                                        [
                                        ScanPartitioner(4),
                                        ForEachBlockPass([
                                            UpdateDataPass("run_ensemble", True),
                                            UpdateDataPass("ensemble_name", qasm_name),
                                            LEAPSynthesisPass(
                                                success_threshold=1e-3,
                                                max_solutions=10,
                                                max_layer=-1
                                            ),
                                            DiversifyEnsemblePass(success_threshold=2e-3),
                                            GenerateProbabilitiesPass(),
                                            BiasFilterPass(),
                                            # GenerateCircuitSamplerPass(combine_sub_blocks=False)
                                        ]),
                                        # GenerateCircuitSamplerPass(combine_sub_blocks=True)
                                        ], 
                                        request_data=True)

    num_partitions = partitioned_circ.num_operations
    print("Split into ", num_partitions, " partitions.")

    # Print Num CNOTs per block

    for op in partitioned_circ.operations():
        b_circ: Circuit = op.gate._circuit
        print(b_circ.gate_counts)


    all_block_data = data[ForEachBlockPass.key][-1]

    for i, b_data in enumerate(all_block_data):
        # If ensemble circuits is still in data, we have succesfully
        # generated a convex ensemble
        if not b_data["use_ensemble"]:
            print(f"Block {i} was filtered out. Defaulting to the original circuit.")