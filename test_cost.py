from bqskit.ir import Circuit
from bqskit.qis import UnitaryMatrix

from bqskit.ir.opt.cost.functions import HilbertSchmidtResidualsGenerator



un_1 = UnitaryMatrix.random(3)
un_2 = UnitaryMatrix.random(3)

frob_dist = un_1.get_frobenius_distance_from(un_2)
norm_frob_dist = un_1.get_norm_frobenius_distance_from(un_2)

cost = HilbertSchmidtResidualsGenerator()

circ = Circuit.from_unitary(un_1)

dist = cost.calc_cost(circ, un_2)


print(frob_dist, norm_frob_dist, dist)