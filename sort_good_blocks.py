import os
from bqskit.ir import Circuit
from bqskit.ir.gates import CNOTGate
from sys import argv

def process_files(circ_name: str, input_folder, good_output_folder, bad_output_folder):
    for filename in os.listdir(input_folder):
        if filename.startswith('block_') and filename.endswith('.qasm'):
            file_path = os.path.join(input_folder, filename)
            circuit = Circuit.from_file(file_path)
            block_num = filename.split('.')[0].split('_')[-1]

            cnot_count = circuit.count(CNOTGate())
            output_filename = f"{circ_name}_{block_num}.qasm"
            if cnot_count > 25:
                output_path = os.path.join(good_output_folder, output_filename)
            else:
                output_path = os.path.join(bad_output_folder, output_filename)
            circuit.save(output_path)


circ_name = argv[1]
input_folder = f"/pscratch/sd/j/jkalloor/bqskit/fixed_block_checkpoints_min/{circ_name}_0_3_8_3"
good_output_folder = 'good_blocks'
bad_output_folder = 'bad_blocks'
process_files(circ_name, input_folder, good_output_folder, bad_output_folder)