from __future__ import annotations
from typing import NamedTuple

from bqskit.compiler.passes.partitioning.quick import QuickPartitioner
from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import CircuitGate
from bqskit.ir.gates import CNOTGate
from bqskit.ir.gates import IdentityGate
from bqskit.ir.gates import TaggedGate
from bqskit.ir.gates.constant.unitary import ConstantUnitaryGate
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix


"""Test run with a linear topology."""
#     0  1  2  3  4        #########
# 0 --o-----o--P--P--    --#-o---o-#-----#######--
# 1 --x--o--x--o-----    --#-x-o-x-#######-o---#--
# 2 -----x--o--x--o-- => --#---x---#---o-#-x-o-#--
# 3 --o--P--x--P--x--    --#########-o-x-#---x-#--
# 4 --x-----------P--    ----------#-x---#######--
#                                  #######

num_q = 5
circ = Circuit(num_q)
circ.append_gate(CNOTGate(), [0, 1])
circ.append_gate(CNOTGate(), [3, 4])
circ.append_gate(CNOTGate(), [1, 2])
circ.append_gate(CNOTGate(), [0, 1])
circ.append_gate(CNOTGate(), [2, 3])
circ.append_gate(CNOTGate(), [1, 2])
circ.append_gate(CNOTGate(), [2, 3])
utry = circ.get_unitary()
g = QuickPartitioner(3)
g.run(circ, {})

g.draw_graph()

assert len(circ) == 3
assert all(isinstance(op.gate, CircuitGate) for op in circ)
placeholder_gate = TaggedGate(IdentityGate(1), '__fold_placeholder__')
assert all(op.gate._circuit.count(placeholder_gate) == 0 for op in circ)  # type: ignore  # noqa
assert circ.get_unitary() == utry
for cycle_index in range(circ.get_num_cycles()):
    assert not circ._is_cycle_idle(cycle_index)