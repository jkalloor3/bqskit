from bqskit.ir.circuit import Circuit
# Generate a super ensemble for some error bounds
from bqskit.ir.gates import CNOTGate
from bqskit.qis import UnitaryMatrix
import os
import glob
import pandas as pd

from util import load_block

from bqskit.ir.opt.cost.functions import HilbertSchmidtCostGenerator

def get_numpy_circ(circ: Circuit):
    return circ.get_unitary().numpy

def get_numpy(unitary: UnitaryMatrix):
    return unitary.numpy

def get_distance(circ: Circuit):
    global target
    cost_1 = HilbertSchmidtCostGenerator().calc_cost(circ, target)
    # assert(np.allclose(cost_1, cost_2))
    return cost_1

def get_cnot_count(circ: Circuit):
    return circ.count(CNOTGate())


def get_counts(circ_name, block_num, tol, num_unique_circs):
    basic_circ = load_block(circ_name, block_num)
    circ = Circuit.from_file(basic_circ)
    orig_count = get_cnot_count(circ)
    
    csv_file = f"block_checkpoints_nisq_0/{circ_name}_{block_num}_{tol}_{num_unique_circs}/data_try1.csv"
    if not os.path.exists(csv_file):
        csv_file = f"block_checkpoints_nisq_0/{circ_name}_{block_num}_{float(tol)}_{num_unique_circs}/data_try1.csv"
    
    if not os.path.exists(csv_file):
        # print(f"{csv_file} does not exist.")
        return 0, 0

    df = pd.read_csv(csv_file)
    min_avg_count = min(df["Avg. CNOT Count"])

    return orig_count, min_avg_count



# Circ 
if __name__ == '__main__':
    
    circs = ["adder9", "heisenberg7", "qae11", "qft_8", "qft_12", "qpe12", "shor_12", "tfxy_6"]

    for circ in circs:
        circ_files = glob.glob(f"good_blocks/{circ}_*.qasm")
        block_nums = [file.split('_')[-1].split('.')[0] for file in circ_files]
        for block_num in block_nums:
            for tol in [0,1,2,3,4,5]:
                for unique_circs in [250]:
                    orig_count, min_avg_count = get_counts(circ, block_num, tol, unique_circs)
                    if orig_count > 0:
                        print(f"{circ}_{block_num}, Tol: {tol} --- {orig_count},  {min_avg_count}", flush=True)