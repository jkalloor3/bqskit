from bqskit.compiler.passdata import PassData
from bqskit.ir.circuit import Circuit
from sys import argv
from bqskit import compile
import numpy as np
# Generate a super ensemble for some error bounds
from bqskit.passes import ForEachBlockPass, LEAPSynthesisPass, WhileLoopPass, ScanPartitioner, PassPredicate
from util import JiggleCircPass, GetErrorsPass
from bqskit.ir.gates import GlobalPhaseGate, ConstantUnitaryGate

from bqskit.qis.unitary import UnitaryMatrix
from bqskit.ir.opt.cost.functions import HilbertSchmidtResidualsGenerator
from bqskit.compiler.compiler import Compiler, WorkflowLike

def cost_4(utry: UnitaryMatrix, target: UnitaryMatrix):
    '''
    Calculates the normalized Frobenius distance between two unitaries
    '''
    diff = utry - target
    # This is Frob(u - v)
    cost = np.sqrt(np.real(np.trace(diff @ diff.conj().T)))

    N = utry.shape[0]
    cost = cost / np.sqrt(2 * N)

    # This quantity should be less than HS distance as defined by 
    # Quest Paper 
    return cost

def cost_4_stepthrough(utry: UnitaryMatrix, target: UnitaryMatrix):
    '''
    Calculates the normalized Frobenius distance between two unitaries
    and prints helpful information
    '''
    diff = utry - target
    N = utry.shape[0]

    a = np.trace(diff @ diff.conj().T)
    b = np.einsum('ij,ij->', diff, diff.conj())

    print("Trace: ", a)
    print("Einsum: ", b)

    cost = np.sqrt(np.real(a))
    print("Square root of real: ", cost)

    # Take 2
    u_u_dag = utry @ utry.conj().T
    print("Trace of UU^dagger: ", np.trace(u_u_dag))

    v_v_dag = target @ target.conj().T

    print("Trace of VV^dagger: ", np.trace(v_v_dag))

    trace_2 = np.trace(u_u_dag) + np.trace(v_v_dag) - 2 * np.real(np.trace(utry @ target.conj().T))

    print("Trace 2: ", trace_2)
    print("Frob Cost 2: ", np.sqrt(trace_2))

    # This is Frob(u - v)
    cost = cost / np.sqrt(2 * N)

    print("Final Cost: ", cost)

    # This quantity should be less than HS distance as defined by 
    # Quest Paper 
    return cost


def frob_cost(utry: UnitaryMatrix, target: UnitaryMatrix):
    diff = utry - target
    cost = np.sqrt(np.real(np.trace(diff @ diff.conj().T)))
    return cost

def HS_cost_2(utry: UnitaryMatrix, target: UnitaryMatrix):
    inside = np.abs(np.trace(utry @ target.conj().T))
    N = utry.shape[0]
    cost = np.sqrt(1 - (inside / N))
    return cost

def cost_5(utry: UnitaryMatrix, target: UnitaryMatrix):
    N = utry.shape[0]
    b = np.einsum('ij,ij->', utry, target.conj())
    # Replace real with abs
    trace_2 = 2 * N - 2 * np.abs(b)
    cost = np.sqrt(trace_2)
    # This is Frob(u - v)
    cost = cost / np.sqrt(2 * N)
    return cost
 

class WhilePredicate(PassPredicate):

    def __init__(self, max_ind: 20):
        self.max_ind = max_ind

    def get_truth_value(self, circuit: Circuit, data: PassData) -> bool:
        if "ind" not in data:
            data["ind"] = 0
        return data["ind"] < self.max_ind


def get_noisy_unitary(unitary: UnitaryMatrix, noise: float = 0.2, global_phase: bool = True):
    N = unitary.shape[0]
    modified_unitary = unitary.numpy + np.random.normal(0, noise, (N, N))
    mod_unitary = UnitaryMatrix.closest_to(modified_unitary)

    if global_phase:
        random_angle = np.random.uniform(0, 2 * np.pi)
        random_phase = np.exp(-1j * random_angle)
    else:
        random_phase = 1

    return mod_unitary * random_phase


def test_1():
    # Test upperbound of Frobenius distance (test is in GetErrorsPass)
    initial_circ = Circuit.from_file("ensemble_benchmarks/hubbard_4.qasm")

    leap_pass = LEAPSynthesisPass(success_threshold=1e-1, cost=HilbertSchmidtResidualsGenerator())

    workflow = [
        ScanPartitioner(3),
        WhileLoopPass(
            condition=WhilePredicate(500),
            loop_body= [
                ForEachBlockPass([
                    JiggleCircPass(),
                    # leap_pass
                ]),
                GetErrorsPass()
            ]
        )
    ]
    compiler = Compiler(num_workers=40)

    compiler.compile(initial_circ, workflow)

def test_2(qudit_size: int = 4):
    # Take a unitary, modify it randomly and track the distance
    initial_unitary = UnitaryMatrix.random(qudit_size)
    N = initial_unitary.shape[0]
    dist_1s = []
    dist_2s = []
    dist_3s = []
    for _ in range(10000):
        modified_unitary = initial_unitary.numpy + np.random.normal(0, 0.2, (N, N))
        mod_unitary = UnitaryMatrix.closest_to(modified_unitary)

        mod_circ = Circuit.from_unitary(mod_unitary)

        dist_1 = HilbertSchmidtResidualsGenerator().calc_cost(mod_circ, initial_unitary)
        dist_2 = mod_unitary.get_frobenius_distance(initial_unitary)
        dist_3 = cost_4(mod_unitary, initial_unitary)

        dist_1s.append(dist_1)
        dist_2s.append(dist_2)
        dist_3s.append(dist_3)


    np.set_printoptions(precision=3, threshold=10000, linewidth=1000)
    is_greater_than = np.array(dist_1s) > np.array(dist_3s)
    print(is_greater_than.all())


def test_3(qudit_size: int = 4):
    # Take a unitary, modify it randomly and track the distance
    initial_unitary = UnitaryMatrix.random(qudit_size)
    N = initial_unitary.shape[0]
    diffs_1 = []
    diffs_2 = []
    for _ in range(10000):
        # Add random global phase
        mod_circ = Circuit.from_unitary(initial_unitary)
        gp_circ = mod_circ.copy()
        random_angle = np.random.uniform(0, 2 * np.pi)
        random_phase = np.exp(-1j * random_angle)
        gp_circ.append_gate(GlobalPhaseGate(global_phase=random_phase), (0,))

        dist_1 = HilbertSchmidtResidualsGenerator().calc_cost(mod_circ, initial_unitary)
        dist_2 = HilbertSchmidtResidualsGenerator().calc_cost(gp_circ, initial_unitary)

        diffs_1.append(dist_1 - dist_2)

        dist_1 = cost_4(mod_circ.get_unitary(), initial_unitary)
        dist_2 = cost_4(gp_circ.get_unitary(), initial_unitary)
        diffs_2.append(dist_1 - dist_2)

    print(np.max(np.abs(diffs_1)))
    print(np.max(np.abs(diffs_2)))

def test_4(qudit_size: int = 4):
    # Take a unitary, modify it randomly and track the distance
    initial_unitary = UnitaryMatrix.random(qudit_size)
    N = initial_unitary.shape[0]
    num_fails_1 = 0
    num_fails_2 = 0
    for _ in range(10000):
        # Add noise and random global phase
        modified_unitary = initial_unitary.numpy + np.random.normal(0, 0.2, (N, N))
        mod_unitary = UnitaryMatrix.closest_to(modified_unitary)
        mod_circ = Circuit.from_unitary(mod_unitary)

        # Add random global phase
        gp_circ = mod_circ.copy()
        random_angle = np.random.uniform(0, 2 * np.pi)
        random_phase = np.exp(-1j * random_angle)
        gp_circ.append_gate(GlobalPhaseGate(global_phase=random_phase), (0,))

        dist_1 = HilbertSchmidtResidualsGenerator().calc_cost(mod_circ, initial_unitary)
        dist_2 = HilbertSchmidtResidualsGenerator().calc_cost(gp_circ, initial_unitary)
        dist_3 = cost_4(mod_unitary, initial_unitary)
        dist_4 = cost_4(mod_unitary * random_phase, initial_unitary)

        if dist_1 < dist_3:
            num_fails_1 += 1
        
        if dist_2 < dist_4:
            num_fails_2 += 1
            # print("Dist 2: ", dist_2)
            # cost_4_stepthrough(gp_circ.get_unitary(), initial_unitary)
            # return

    print(num_fails_1 / 10000)
    print(num_fails_2 / 10000)

def HS_cost(utry: UnitaryMatrix, target: UnitaryMatrix):
    circ = Circuit.from_unitary(utry)
    return HilbertSchmidtResidualsGenerator().calc_cost(circ, target)

def test_5(num_qudits: int = 4):
    '''
    Tests if normalized and normal frobenius distances are upperbounded
    by the sum of their distances
    '''
    factors_1 = []
    factors_2 = []
    num_unitariess = []
    for _ in range(10000):
        num_unitaries = np.random.randint(20, 50)
        num_unitariess.append(num_unitaries)
        # Create 20 random unitaries of different sizes < num_qudits
        num_qubits = np.random.randint(2, num_qudits + 1, num_unitaries)
        unitaries = [UnitaryMatrix.random(i) for i in num_qubits]
        # Add noise and random global phase to each unitary
        mod_unitaries = [get_noisy_unitary(unitary, noise=0.00001) for unitary in unitaries]

        # Calculate the distance between each noisy unitary and the original
        dists_1 = [frob_cost(mod_unitary, unitary) for mod_unitary, unitary in zip(mod_unitaries, unitaries)]
        dists_2 = [cost_4(mod_unitary, unitary) for mod_unitary, unitary in zip(mod_unitaries, unitaries)]

        upper_bound_1 = np.sum(dists_1)
        upper_bound_2 = np.sum(dists_2)

        # Randomly add unitaries to a circuit
        rand_locs = [tuple(np.random.choice(num_qudits, i, replace=False)) for i in num_qubits]

        circ_1 = Circuit(num_qudits=num_qudits)
        circ_2 = Circuit(num_qudits=num_qudits)

        for loc, unitary, mod_unitary in zip(rand_locs, unitaries, mod_unitaries):
            circ_1.append_gate(ConstantUnitaryGate(unitary), loc)
            circ_2.append_gate(ConstantUnitaryGate(mod_unitary), loc)
        
        final_unitary = circ_1.get_unitary()
        noisy_unitary = circ_2.get_unitary()

        actual_dist_1 = frob_cost(noisy_unitary, final_unitary)
        actual_dist_2 = cost_4(noisy_unitary, final_unitary)

        if upper_bound_1 < actual_dist_1:
            print("Upper bound 1 failed")
            print(upper_bound_1, actual_dist_1)

        if upper_bound_2 < actual_dist_2:
            print("Upper bound 2 failed")
            print(upper_bound_2, actual_dist_2)

        # print(upper_bound_1, actual_dist_1)
        # print("Frobenius Overestimate Factor: ", upper_bound_1 / actual_dist_1 , "Num Unitaries: ", num_unitaries)
        # print("Normalized Frobenius Overestimate Factor: ", upper_bound_2 / actual_dist_2 , "Num Unitaries: ", num_unitaries)
        factors_1.append(upper_bound_1 / actual_dist_1)
        factors_2.append(upper_bound_2 / actual_dist_2)
    
    # Scatter plot factors vs num unitaries

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    print("Average Overestimate Factor for Frobenius: ", np.mean(factors_1))
    print("Average Overestimate Factor for Normalized Frobenius: ", np.mean(factors_2))

    print("Median Overestimate Factor for Frobenius: ", np.median(factors_1))
    print("Median Overestimate Factor for Normalized Frobenius: ", np.median(factors_2))

    ax.scatter(factors_1, factors_2)

    ax.set_xlabel("Overestimate Factor Frobenius")
    ax.set_ylabel("Overestimate Factor Normalized")

    ax.set_ybound(0, 2 * np.median(factors_2))
    ax.set_xbound(0, 2 * np.median(factors_1))
    ax.legend()

    fig.savefig(f"overestimate_factors_{num_qudits}.png")


def test_7(num_qudits: int = 4):
    '''
    Find normalied distance that is closest to cost_4 after
    applying a global phase correction
    '''
    num_points = 20
    dists_norm = []
    dists_hs = []
    dists_hs_2 = []
    dists_norm_abs = []

    for _ in range(num_points):
        target = UnitaryMatrix.random(num_qudits)
        # Do not add global phase
        noise_level = np.random.uniform(0, 0.2)
        noisy_unitary = get_noisy_unitary(target, noise=noise_level, global_phase=False)

        # Correct global phase
        global_phase_correction = target.get_target_correction_factor(noisy_unitary)

        dists_norm.append(cost_4(noisy_unitary * global_phase_correction, target))
        dists_hs.append(HS_cost(noisy_unitary * global_phase_correction, target))
        dists_hs_2.append(HS_cost_2(noisy_unitary, target))
        dists_norm_abs.append(cost_5(noisy_unitary, target))

    # Now get correlation to dists_norm
    dists_norm = np.array(dists_norm)
    dists_hs = np.array(dists_hs)
    dists_hs_2 = np.array(dists_hs_2)
    dists_norm_abs = np.array(dists_norm_abs)

    print("Correlation to HS: ", np.corrcoef(dists_norm, dists_hs)[0, 1])
    print("Correlation to HS 2: ", np.corrcoef(dists_norm, dists_hs_2)[0, 1])
    print("Correlation to HS 3: ", np.corrcoef(dists_norm, dists_norm_abs)[0, 1])


def test_6(num_qudits: int = 4):
    '''
    Tests if normalized and normal frobenius distances are upperbounded
    by the sum of their distances.
    Test in case of only changing 1 unitary
    '''
    factors_1 = []
    factors_2 = []
    num_unitariess = []
    for _ in range(10000):
        num_unitaries = np.random.randint(20, 50)
        num_unitariess.append(num_unitaries)
        # Create 20 random unitaries of different sizes < num_qudits
        num_qubits = np.random.randint(2, num_qudits, num_unitaries)
        unitaries = [UnitaryMatrix.random(i) for i in num_qubits]
        # Add noise and random global phase to each unitary
        mod_unitaries = [UnitaryMatrix(u) for u in unitaries]
        rand_ind = np.random.randint(0, num_unitaries)
        mod_unitaries[rand_ind] = get_noisy_unitary(unitaries[rand_ind], noise=0.1)

        # Calculate the distance between each noisy unitary and the original
        dists_1 = [frob_cost(mod_unitary, unitary) for mod_unitary, unitary in zip(mod_unitaries, unitaries)]
        dists_2 = [cost_4(mod_unitary, unitary) for mod_unitary, unitary in zip(mod_unitaries, unitaries)]

        upper_bound_1 = np.sum(dists_1)
        upper_bound_2 = np.sum(dists_2)

        # Randomly add unitaries to a circuit
        rand_locs = [tuple(np.random.choice(num_qudits, i, replace=False)) for i in num_qubits]

        circ_1 = Circuit(num_qudits=num_qudits)
        circ_2 = Circuit(num_qudits=num_qudits)

        for loc, unitary, mod_unitary in zip(rand_locs, unitaries, mod_unitaries):
            circ_1.append_gate(ConstantUnitaryGate(unitary), loc)
            circ_2.append_gate(ConstantUnitaryGate(mod_unitary), loc)
        
        final_unitary = circ_1.get_unitary()
        noisy_unitary = circ_2.get_unitary()

        actual_dist_1 = frob_cost(noisy_unitary, final_unitary)
        actual_dist_2 = cost_4(noisy_unitary, final_unitary)

        # if upper_bound_1 < actual_dist_1:
        #     print("Upper bound 1 failed")
        #     print(upper_bound_1, actual_dist_1)

        if (upper_bound_2 + 0.01) <= actual_dist_2:
            print("Upper bound 2 failed")
            print(upper_bound_2, actual_dist_2)

        # print(upper_bound_1, actual_dist_1)
        # print("Frobenius Overestimate Factor: ", upper_bound_1 / actual_dist_1 , "Num Unitaries: ", num_unitaries)
        # print("Normalized Frobenius Overestimate Factor: ", upper_bound_2 / actual_dist_2 , "Num Unitaries: ", num_unitaries)
        factors_1.append(upper_bound_1 / actual_dist_1)
        factors_2.append(upper_bound_2 / actual_dist_2)
    
    # Scatter plot factors vs num unitaries

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    print("Average Overestimate Factor for Frobenius: ", np.mean(factors_1))
    print("Average Overestimate Factor for Normalized Frobenius: ", np.mean(factors_2))

    print("Median Overestimate Factor for Frobenius: ", np.median(factors_1))
    print("Median Overestimate Factor for Normalized Frobenius: ", np.median(factors_2))

    ax.scatter(factors_1, factors_2)

    ax.set_xlabel("Overestimate Factor Frobenius")
    ax.set_ylabel("Overestimate Factor Normalized")

    ax.set_ybound(0, 2 * np.median(factors_2))
    ax.set_xbound(0, 2 * np.median(factors_1))
    ax.legend()

    fig.savefig(f"overestimate_factors_{num_qudits}_single_unitary.png")

from sys import argv

if __name__ == '__main__':
    # test_2(4)
    # test_2(5)
    # test_2(6)
    # test_2(7)
    # test_1()
    # test_3(4)
    # test_3(5)
    # test_3(6)
    # test_3(7)
    # test_4(2)
    num_qudits = int(argv[1])
    # test_5(num_qudits)
    # test_6(num_qudits)
    test_7(num_qudits)