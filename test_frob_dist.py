from bqskit.ir.circuit import Circuit
from sys import argv
from bqskit import compile
import numpy as np
# Generate a super ensemble for some error bounds
from bqskit.passes import *
from util import JiggleCircPass, GetErrorsPass

from bqskit.qis.unitary import UnitaryMatrix
from bqskit.ir.opt.cost.functions import HilbertSchmidtCostGenerator
from bqskit.compiler.compiler import Compiler, WorkflowLike

initial_circ = Circuit.from_file("ensemble_benchmarks/hubbard_4.qasm")

workflow = [
    ScanPartitioner(3),
    ForEachBlockPass([
        JiggleCircPass(),
    ]),
    GetErrorsPass()
]

# un_1 = initial_circ.get_unitary()

# new_params = initial_circ.params.copy()
# new_params[0] = -3

# target = initial_circ.get_unitary(new_params)

# print(un_1.get_frobenius_distance(target))
# print(un_1.get_distance_from(target))

# cost = HilbertSchmidtCostGenerator()

# print(cost.calc_cost(initial_circ, target))

compiler = Compiler(num_workers=256)

compiled_circuit = compiler.compile(initial_circ, workflow)