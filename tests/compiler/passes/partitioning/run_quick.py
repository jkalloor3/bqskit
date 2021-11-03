from __future__ import annotations
from typing import NamedTuple
import argparse

from bqskit.passes.partitioning.quick import QuickPartitioner
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import CircuitGate
from bqskit.ir.gates import CNOTGate
from bqskit.ir.gates import IdentityGate
from bqskit.ir.gates import TaggedGate
from bqskit.ir.gates.constant.unitary import ConstantUnitaryGate
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.ir.lang.qasm2.qasm2 import OPENQASM2Language


"""Test run with a linear topology."""
#     0  1  2  3  4        #########
# 0 --o-----o--P--P--    --#-o---o-#-----#######--
# 1 --x--o--x--o-----    --#-x-o-x-#######-o---#--
# 2 -----x--o--x--o-- => --#---x---#---o-#-x-o-#--
# 3 --o--P--x--P--x--    --#########-o-x-#---x-#--
# 4 --x-----------P--    ----------#-x---#######--
#                                  #######
parser = argparse.ArgumentParser(
    description="Run subtopoloy aware synthesis"
                " based on the hybrid logical-physical topology scheme"
)
parser.add_argument("qasm_file", type=str, help="file to synthesize")


args = parser.parse_args()
# num_q = 5
# circ = Circuit(num_q)
# circ.append_gate(CNOTGate(), [0, 1])
# circ.append_gate(CNOTGate(), [3, 4])
# circ.append_gate(CNOTGate(), [1, 2])
# circ.append_gate(CNOTGate(), [0, 1])
# circ.append_gate(CNOTGate(), [2, 3])
# circ.append_gate(CNOTGate(), [1, 2])
# circ.append_gate(CNOTGate(), [2, 3])
# utry = circ.get_unitary()
# QuickPartitioner(3).run(circ, {})

with open(args.qasm_file, 'r') as f:
    circ = OPENQASM2Language().decode(f.read())

QuickPartitioner(4).run(circ, {})

# assert len(circ) == 3
# assert all(isinstance(op.gate, CircuitGate) for op in circ)
# placeholder_gate = TaggedGate(IdentityGate(1), '__fold_placeholder__')
# assert all(op.gate._circuit.count(placeholder_gate) == 0 for op in circ)  # type: ignore  # noqa
# assert circ.get_unitary() == utry
# for cycle_index in range(circ.num_cycles):
#     assert not circ._is_cycle_idle(cycle_index)