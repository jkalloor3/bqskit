import os
from bqskit.ir import Circuit
from bqskit.ir.gates import CNOTGate
from sys import argv
from bqskit.passes import ScanPartitioner, ForEachBlockPass, ExtractMeasurements
from util import WriteQasmPass
from bqskit.compiler import Compiler
from bqskit import enable_logging
import time

enable_logging(True)

block_save_dir = "block_qasms"

partition_workflow = [
    ExtractMeasurements(),
    ScanPartitioner(8),
    ForEachBlockPass([
        WriteQasmPass(block_save_dir)
    ])
]

compiler = Compiler(num_workers=16)

def process_files(circ_name: str, input_folder, good_output_folder, bad_output_folder):
    filename = f"{circ_name}.qasm"
    circ = Circuit.from_file(os.path.join(input_folder, filename))
    print("Running Partitioner on: ", filename)
    compiler.compile(circ, partition_workflow)
    print("Sleeping for 5 seconds to allow for file writes")
    time.sleep(5)
    for filename in os.listdir(block_save_dir):
        if filename.endswith('.qasm'):
            file_path = os.path.join(block_save_dir, filename)
            circuit = Circuit.from_file(file_path)
            block_num = filename.split('.')[0].split('_')[-1]

            cnot_count = circuit.count(CNOTGate())
            output_filename = f"{circ_name}_{block_num}.qasm"
            if cnot_count > 25:
                output_path = os.path.join(good_output_folder, output_filename)
            else:
                output_path = os.path.join(bad_output_folder, output_filename)
            circuit.save(output_path)
            os.unlink(file_path)
            time.sleep(1)


# circ_name = argv[1]
circ_names = ["qpe_14"]
# circ_names = ["pricingcall_indep_qiskit_13", "pricingput_indep_qiskit_13", "qaoa_indep_qiskit_11", "qwalk-noancilla_indep_qiskit_8"]
# input_folder = f"/pscratch/sd/j/jkalloor/bqskit/MQTBench"
input_folder = "ensemble_benchmarks"
good_output_folder = 'good_blocks'
bad_output_folder = 'bad_blocks'
for circ_name in circ_names:
    process_files(circ_name, input_folder, good_output_folder, bad_output_folder)