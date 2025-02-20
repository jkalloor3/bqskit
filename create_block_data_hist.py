from bqskit.ir.circuit import Circuit
from sys import argv
from bqskit.compiler.compiler import Compiler
from bqskit.passes import ScanPartitioner, ExtendBlockSizePass, CheckpointRestartPass, ForEachBlockPass
# Generate a super ensemble for some error bounds

from util import AnalyzeBlockPass, WriteQasmPass, MakeHistogramPass
from util import WriteQasmPass

from util import load_circuit
import os
import glob
import pickle
import pandas as pd

checkpoint_dir = "block_histograms/"

def partition(circ_name: str) -> None:
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

    leap_workflow = [
        CheckpointRestartPass(checkpoint_dir, 
                                default_passes=slow_partitioner_passes),
        ForEachBlockPass(
            [
                MakeHistogramPass(),
            ],
        ),
        MakeHistogramPass(),
    ]
    num_workers = 128
    compiler = Compiler(num_workers=3)
    compiler.compile(circ, workflow=leap_workflow, request_data=True)
    return 

def get_csv_data(file_name: str) -> list:
    data = pd.read_csv(file_name, header=0)
    # print(data['Ratio'])
    return min(data['Ratio'])

def get_data(file_name: str) -> tuple[list, list, list, list]:
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
    return data['2Q Count'], data['Depth'], data['Free Params'], data['Widths']
    
def create_small_block_histogram(circ_name = ""):
    dirs = glob.glob(f"block_histograms/{circ_name}*")
    print(dirs)
    if len(dirs) == 0:
        return
    all_data = {}
    all_data["2Q Count"] = []
    all_data["Depth"] = []
    all_data["Free Params"] = []
    all_data["Widths"] = []
    for folder_path in dirs:
        if os.path.isdir(folder_path):
            data_files = glob.glob(os.path.join(folder_path, 'block*.data'))
            for data_file in data_files:
                counts, depths, params, widths = get_data(data_file)
                print("File: ", data_file)
                print("Total Counts: ", len(counts))
                all_data["2Q Count"].extend(counts)
                all_data["Depth"].extend(depths)
                all_data["Free Params"].extend(params)
                all_data["Widths"].extend(widths)
    
    MakeHistogramPass.create_histogram(all_data, f'{circ_name}_small_block_histograms.png')
    
def create_large_block_histogram(circ_name = ""):
    dirs = glob.glob(f"block_histograms/{circ_name}*")
    if len(dirs) == 0:
        return
    all_data = {}
    all_data["2Q Count"] = []
    all_data["Depth"] = []
    all_data["Free Params"] = []
    all_data["Widths"] = []
    for folder_path in dirs:
        if os.path.isdir(folder_path):
            data_file = os.path.join(folder_path, 'data.data')
            counts, depths, params, widths = get_data(data_file)
            all_data["2Q Count"].extend(counts)
            all_data["Depth"].extend(depths)
            all_data["Free Params"].extend(params)
            all_data["Widths"].extend(widths)
    
    MakeHistogramPass.create_histogram( all_data,  f'{circ_name}_large_block_histograms.png')

def create_ratio_histogram(full_checkpoint_dir: str):
    all_data = {}
    all_data["Bias Reduction Ratio"] = []
    for folder_name in os.listdir(full_checkpoint_dir):
        # run = False
        # for inc in includes:
        #     if folder_name.startswith(inc):
        #         run = True
        # if not run:
        #     continue
        folder_path = os.path.join(full_checkpoint_dir, folder_name)
        # print(folder_path)
        if os.path.isdir(folder_path):
            data_file = os.path.join(folder_path, 'data_try1.csv')
            if os.path.exists(data_file):
                print(data_file)
                min_ratio = get_csv_data(data_file)
                all_data["Bias Reduction Ratio"].append(min_ratio)
    
    MakeHistogramPass.create_histogram( all_data,  'good_ratio_histogram.png', False)


if __name__ == '__main__':
    global target
    circ_name = argv[1]
    # partition(circ_name)
    create_small_block_histogram(circ_name)
    create_large_block_histogram(circ_name=circ_name)
    # cliff_t_dir = "/home/jkalloor/bqskit/block_checkpoints_clifft"
    # nisq_dir = "/home/jkalloor/bqskit/bqskit/block_checkpoints_nisq_0"
    # create_ratio_histogram(cliff_t_dir)
