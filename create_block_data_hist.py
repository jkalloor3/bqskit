from bqskit.ir.circuit import Circuit
from sys import argv
from bqskit.compiler.compiler import Compiler
# Generate a super ensemble for some error bounds
from bqskit.passes import *
from bqskit.ir.gates import CNOTGate

from util import AnalyzeBlockPass, WriteQasmPass, MakeHistogramPass
from util import WriteQasmPass

from util import load_circuit
import os
import glob
import pickle

checkpoint_dir = "block_histograms/"


def get_shortest_circuits(circ_name: str) -> list[Circuit]:
    circ = load_circuit(circ_name)

    print("Original Gate Counts: ", circ.gate_counts, flush=True)
    
        
    big_block_size = 8
    small_block_size = 3
    checkpoint_dir = f"block_histograms/{circ_name}_{big_block_size}_{small_block_size}/"

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
    else:
        partitioner_passes = slow_partitioner_passes

    leap_workflow = [
        CheckpointRestartPass(checkpoint_dir, 
                                default_passes=partitioner_passes),
        ForEachBlockPass(
            [
                WriteQasmPass(),
                AnalyzeBlockPass(CNOTGate()),
                ForEachBlockPass(
                    [
                        WriteQasmPass(),
                        AnalyzeBlockPass(CNOTGate()),
                    ]
                ),
                MakeHistogramPass(),
            ]
        ),
        MakeHistogramPass(),
    ]
    num_workers = 20
    compiler = Compiler(num_workers=num_workers)
    # target = circ.get_unitary()
    out_circ, data = compiler.compile(circ, workflow=leap_workflow, request_data=True)
    return 


def get_data(file_name: str) -> tuple[list, list, list, list]:
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
    return data['2q_counts'], data['depths'], data['params'], data['widths']

def create_small_block_histogram():
    all_counts = []
    all_depths = []
    all_params = []
    all_widths = []
    for folder_name in os.listdir(checkpoint_dir):
        folder_path = os.path.join(checkpoint_dir, folder_name)
        if os.path.isdir(folder_path):
            data_files = glob.glob(os.path.join(folder_path, 'block*.data'))
            for data_file in data_files:
                counts, depths, params, widths = get_data(data_file)
                all_counts.extend(counts)
                all_depths.extend(depths)
                all_params.extend(params)
                all_widths.extend(widths)
    
    MakeHistogramPass.create_histogram(
        [all_counts, all_depths, all_params, all_widths], 
        ['2Q Count', 'Depth', 'Params', 'Width'], 
        'small_block_histograms.png')
    
def create_large_block_histogram():
    all_counts = []
    all_depths = []
    all_params = []
    all_widths = []
    for folder_name in os.listdir(checkpoint_dir):
        folder_path = os.path.join(checkpoint_dir, folder_name)
        if os.path.isdir(folder_path):
            data_file = os.path.join(folder_path, 'data.data')
            counts, depths, params, widths = get_data(data_file)
            all_counts.extend(counts)
            all_depths.extend(depths)
            all_params.extend(params)
            all_widths.extend(widths)
    MakeHistogramPass.create_histogram(
        [all_counts, all_depths, all_params, all_widths], 
        ['2Q Count', 'Depth', 'Params', 'Width'], 
        'large_block_histograms.png')


if __name__ == '__main__':
    global target
    circ_name = argv[1]
    get_shortest_circuits(circ_name)
    # create_large_block_histogram()
    # create_small_block_histogram()