import os
from sys import argv
import pickle
import glob
import numpy as np
from bqskit.ir import Circuit
from bqskit.qis import StateVector

import matplotlib.pyplot as plt
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
from qiskit import QuantumCircuit
from bqskit.ir.gates import GlobalPhaseGate
from bqskit.ext import bqskit_to_qiskit

from util.plot_lib import plot_tvd_convergence

if __name__ == '__main__':
    circ_name = argv[1] if len(argv) > 1 else 'qae11'
    noisy = bool(int(argv[2])) if len(argv) > 2 else False


    fig, axs = plt.subplots(1, 1, figsize=(10, 6))
    plot_tvd_convergence(circ_name, axs, noisy=noisy)

    fig.savefig(f'tvd_convergence_{circ_name}.png', dpi=300)