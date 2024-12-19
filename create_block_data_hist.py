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

includes = ["qft", "adder"]


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
                MakeHistogramPass(),
            ],
        ),
        MakeHistogramPass(),
    ]
    num_workers = 128
    compiler = Compiler(num_workers=num_workers)
    # target = circ.get_unitary()
    out_circ, data = compiler.compile(circ, workflow=leap_workflow, request_data=True)
    return 


def get_data(file_name: str) -> tuple[list, list, list, list]:
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
    return data['2Q Count'], data['Depth'], data['Free Params'], data['Widths']

def create_small_block_histogram():
    all_data = {}
    all_data["2Q Count"] = []
    all_data["Depth"] = []
    all_data["Free Params"] = []
    all_data["Widths"] = []
    for folder_name in os.listdir(checkpoint_dir):
        run = False
        for inc in includes:
            if folder_name.startswith(inc):
                run = True
        if not run:
            continue
        folder_path = os.path.join(checkpoint_dir, folder_name)
        if os.path.isdir(folder_path):
            data_files = glob.glob(os.path.join(folder_path, 'block*.data'))
            for data_file in data_files:
                counts, depths, params, widths = get_data(data_file)
                all_data["2Q Count"].extend(counts)
                all_data["Depth"].extend(depths)
                all_data["Free Params"].extend(params)
                all_data["Widths"].extend(widths)
    
    MakeHistogramPass.create_histogram(all_data, 'small_block_histograms.png')
    
def create_large_block_histogram():
    all_data = {}
    all_data["2Q Count"] = []
    all_data["Depth"] = []
    all_data["Free Params"] = []
    all_data["Widths"] = []
    for folder_name in os.listdir(checkpoint_dir):
        run = False
        for inc in includes:
            if folder_name.startswith(inc):
                run = True
        if not run:
            continue
        folder_path = os.path.join(checkpoint_dir, folder_name)
        if os.path.isdir(folder_path):
            data_file = os.path.join(folder_path, 'data.data')
            counts, depths, params, widths = get_data(data_file)
            all_data["2Q Count"].extend(counts)
            all_data["Depth"].extend(depths)
            all_data["Free Params"].extend(params)
            all_data["Widths"].extend(widths)
    
    MakeHistogramPass.create_histogram( all_data,  'large_block_histograms.png')


if __name__ == '__main__':
    global target
    circ_name = argv[1]
    # get_shortest_circuits(circ_name)
    create_large_block_histogram()
    create_small_block_histogram()