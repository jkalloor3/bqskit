from __future__ import annotations

from bqskit.compiler.machine import MachineModel
from bqskit.compiler.passes.simplepartitioner import SimplePartitioner
from bqskit.ir import Circuit
from bqskit.ir.gates.constant.cx import CNOTGate

#     0  1  2  3           #########
# 0 --o-----o--------    --#-o---o-#-----#######--
# 1 --x--o--x--o-----    --#-x-o-x-#######-o---#--
# 2 -----x--o--x--o-- => --#---x---#---o-#-x-o-#--
# 3 --o-----x-----x--    --#########-o-x-#---x-#--
# 4 --x--------------    ----------#-x---#######--
#                                  #######
num_q = 5
coup_map = {(0, 1), (1, 2), (2, 3), (3, 4)}
circ = Circuit(num_q)
circ.append_gate(CNOTGate(), [0, 1])
circ.append_gate(CNOTGate(), [3, 4])
circ.append_gate(CNOTGate(), [1, 2])
circ.append_gate(CNOTGate(), [0, 1])
circ.append_gate(CNOTGate(), [2, 3])
circ.append_gate(CNOTGate(), [1, 2])
circ.append_gate(CNOTGate(), [2, 3])
mach = MachineModel(num_q, coup_map)
part = SimplePartitioner(mach, 3)

part.run(circ, {})

n = 1000
coup_map = set()
num_qudits = n ** 2

for i in range(0, n):
    for j in range(2, n):
        coup_map.add((i*n + j-1, i*n + j))
for i in range(0,n):
    for j in range(0,n-2):
        coup_map.add((i*n + j, i*n + j+1))
for k in range(0, n*(n-1)):
    coup_map.add((k, k+n))

block_size = 3

mach = MachineModel(num_qudits, coup_map)
part = SimplePartitioner(mach, block_size)
# Get all qubit groups
from time import time
start = time()
groups = part.get_qudit_groups()
stop = time()
print(stop - start)
print(len(groups))