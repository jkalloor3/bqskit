import os
from bqskit.ir import Circuit
from bqskit.ir.gates import CNOTGate
from sys import argv
from bqskit.passes import ScanPartitioner, ForEachBlockPass, ExtractMeasurements
from util import WriteQasmPass
from bqskit.compiler import Compiler
from bqskit import enable_logging
import time
import glob

# enable_logging(True)

block_save_dir = "/pscratch/sd/j/jkalloor/bqskit/block_qasms_{circ_name}/"

compiler = Compiler(num_workers=128)

def partition_workflow(circ_name: str):
    return [
    ExtractMeasurements(),
    ScanPartitioner(8),
    ForEachBlockPass([
        WriteQasmPass(block_save_dir.format(circ_name=circ_name))
    ])
]

def process_files(circ_name: str, circ_file: str):
    circ = Circuit.from_file(circ_file)
    print("Running Partitioner on: ", circ_file, flush=True)
    # compiler.compile(circ, partition_workflow)
    return compiler.submit(circ, partition_workflow(circ_name))

def sort_blocks(circ_name: str, good_output_folder, bad_output_folder):
    save_dir = block_save_dir.format(circ_name=circ_name)
    for filename in os.listdir(save_dir):
        if filename.endswith('.qasm'):
            file_path = os.path.join(save_dir, filename)
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
    
    # Delete block_save_dir
    os.rmdir(block_save_dir.format(circ_name=circ_name))

circ_types = ["*"]
input_folder = f"/pscratch/sd/j/jkalloor/bqskit/QITE_8"
good_output_folder = 'good_blocks'
bad_output_folder = 'bad_blocks'
job_ids = []
for circ_type in circ_types:
    circ_files = glob.glob(os.path.join(input_folder, f"{circ_type}.qasm"))
    circ_names = [circ_file.split('/')[-1].split('.')[0] for circ_file in circ_files]
    circ_data = list(zip(circ_names, circ_files))
    for name, file in circ_data:
        job_ids.append((name, process_files(name, file)))

for name, job_id in job_ids:
    compiler.result(job_id)
    print("Finished: ", name, flush=True)
    sort_blocks(name, good_output_folder, bad_output_folder)