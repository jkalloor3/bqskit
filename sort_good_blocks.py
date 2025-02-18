import os
from bqskit.ir import Circuit
from bqskit.ir.gates import CNOTGate
from sys import argv
from bqskit.passes import ScanPartitioner, ExtractMeasurements, ExtendBlockSizePass, ForEachBlockPass, UnfoldPass
from util import WriteQasmPass
from bqskit.compiler import Compiler
from bqskit import enable_logging
import time
import glob
# from pathlib import Path
import pickle
import shutil

# enable_logging(True)
# input_folder = f"/pscratch/sd/j/jkalloor/bqskit/QITE_8"
input_folder = "/pscratch/sd/j/jkalloor/bqskit/qce23*"
good_output_folder = 'good_blocks'
bad_output_folder = 'bad_blocks'
block_save_dir = "/pscratch/sd/j/jkalloor/bqskit/block_qasms_{circ_name}/"
partitioned_circ_save_file = "/pscratch/sd/j/jkalloor/bqskit/partitioned_circs/{circ_name}.pickle"

compiler = Compiler(num_workers=64)

def partition_workflow(circ_name: str):
    return [
    ExtractMeasurements(),
    ScanPartitioner(3),
    ExtendBlockSizePass(3),
    ScanPartitioner(8),
    ForEachBlockPass([
        UnfoldPass(),
        WriteQasmPass(block_save_dir.format(circ_name=circ_name),
                      write=True)
    ])
]

def process_files(circ_name: str, circ_file: str):
    circ = Circuit.from_file(circ_file)
    print("Running Partitioner on: ", circ_name, circ.num_qudits, flush=True)
    # compiler.compile(circ, partition_workflow)
    return compiler.submit(circ, partition_workflow(circ_name))

def sort_blocks(circ_name: str, good_output_folder, bad_output_folder):
    save_dir = block_save_dir.format(circ_name=circ_name)
    for filename in os.listdir(save_dir):
        if filename.endswith('.qasm'):
            file_path = os.path.join(save_dir, filename)
            circuit = Circuit.from_file(file_path)
            print(circuit.num_qudits, file_path)
            qasm_str = open(file_path, 'r').read()
            block_num = filename.split('.')[0].split('_')[-1]

            # cnot_count = circuit.count(CNOTGate())
            cnot_count = qasm_str.count('cx')
            output_filename = f"{circ_name}_{block_num}.qasm"
            if cnot_count > 25:
                output_path = os.path.join(good_output_folder, output_filename)
            else:
                output_path = os.path.join(bad_output_folder, output_filename)
            # circuit.save(output_path)
            # Move file to new file path
            shutil.move(file_path, output_path)
            if os.path.exists(file_path):
                print(f"Somehow file exists! {circ_name}_{block_num}", flush=True)
                os.unlink(file_path)
            time.sleep(1)
    
    # Delete block_save_dir
    os.rmdir(block_save_dir.format(circ_name=circ_name))

circ_types = ["*add17*", "*mult16*", "qae11", "qae13", "qaoa10", "hhl8"]
job_ids = []
for circ_type in circ_types:
    circ_files = glob.glob(os.path.join(input_folder, f"{circ_type}.qasm"))
    circ_names = [circ_file.split('/')[-1].split('.')[0] for circ_file in circ_files]
    circ_data = list(zip(circ_names, circ_files))
    for name, file in circ_data:
        job_ids.append((name, process_files(name, file)))

print(job_ids, flush=True)

for name, job_id in job_ids:
    if job_id == -1:
        print("Partitioning failed for: ", name, flush=True)
        continue
    out_circ = compiler.result(job_id)
    print("Finished: ", name, flush=True)
    sort_blocks(name, good_output_folder, bad_output_folder)
    pickle.dump(out_circ, open(partitioned_circ_save_file.format(circ_name=name), 'wb'))