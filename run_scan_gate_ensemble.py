import pickle
from bqskit.passes import TreeScanningGateRemovalPass, SetTargetPass
from bqskit.ir import Circuit
from bqskit.ir.gates import CNOTGate
from bqskit.compiler import Compiler

# Function to read input file
def read_input_file(file_path):
    with open(file_path, 'r') as file:
        data = file.read()
    return data

# Function to save ensemble to pickle file
def save_to_pickle(data, output_file):
    with open(output_file, 'wb') as file:
        pickle.dump(data, file)

# Main function
def main(input_file, actual_input_file, output_file):
    # Read the input file
    circ = Circuit.from_file(input_file)
    utry = Circuit.from_file(actual_input_file).get_unitary()

    print("Original Circuit: ", circ.gate_counts)
    # print("Utry Circuit: ", utry)
    
    # Run ScanningGateRemovalPass to generate an ensemble
    sgr_pass = TreeScanningGateRemovalPass(store_all_solutions=True, 
                                       success_threshold=1e-5,
                                       tree_depth=7)
    
    workflow = [
        SetTargetPass(utry),
        sgr_pass
    ]
    compiler = Compiler(num_workers=128)


    out_circ, data = compiler.compile(circ, workflow=workflow, request_data=True)

    # Get the ensemble
    ensemble: list[tuple[Circuit, float]] = data['ensemble']

    print("Ensemble Size: ", len(ensemble))
    cnot_counts = [c[0].count(CNOTGate()) for c in ensemble]
    print("CNOT Counts: ", cnot_counts)
    
    # Save the ensemble to a pickle file
    save_to_pickle(ensemble, output_file)

if __name__ == "__main__":
    input_file = '/pscratch/sd/j/jkalloor/bqskit/fixed_block_checkpoints_min/adder9_0_3_8_3/block_2_decomposed.qasm'
    actual_input_file = '/pscratch/sd/j/jkalloor/bqskit/fixed_block_checkpoints_min/adder9_0_3_8_3/block_2.qasm'
    output_file = '/pscratch/sd/j/jkalloor/bqskit/fixed_block_checkpoints_min/adder9_0_3_8_3/block_2_scan_ensemble.pkl'
    main(input_file, actual_input_file, output_file)