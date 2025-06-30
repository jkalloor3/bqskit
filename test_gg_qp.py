# Now let's see if GenerateProbabilityPass can work on full ensemble
from bqskit.runtime import get_runtime

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from itertools import chain, product
from util.generate_probs_pass import GenerateProbabilityPass
import numpy as np
# import time
from bqskit.ir.gates import CNOTGate, U3Gate
from bqskit.qis import UnitaryMatrix
from math import ceil
from util.gg import get_rz_perturbations, gg_gate_def, GridSynthGate, gridsynth_gates_to_cir
from bqskit.ir.gates import RZGate
from bqskit.passes import ScanPartitioner, SetTargetPass
from bqskit.compiler import Compiler
from bqskit.ir.circuit import Circuit, CircuitGate, CircuitPoint
from util.distance import frobenius_cost, normalized_gp_frob_cost, gp_frobenius_cost
from util.common import load_block, load_ensemble
from util.distance import get_corrected_un, tvd
import csv
from bqskit.ir.lang.qasm2 import OPENQASM2Language
from bqskit.ir.opt.cost.functions import GPNormalizedFrobeniusCostGenerator, GPNormalizedFrobeniusCostGenerator
import cvxpy as cp
import cvxopt
from qpsolvers import solve_qp

circ_name = "qae13"
block_name = "3"
tol = 3.0
ind = 0
small_block = "0"


qasms_file = f"small_block_checkpoints_final_paper_4_clifft_tket/{circ_name}_{block_name}_{tol}/block_{small_block}/ensemble_0_.qasms"
jiggle_file = f"small_block_checkpoints_final_paper_4_clifft_tket/{circ_name}_{block_name}_{tol}/block_{small_block}/ensemble_0_jiggles_.npy"
cache_file = f"small_block_checkpoints_final_paper_4_clifft_tket/{circ_name}_{block_name}_{tol}/block_{small_block}/ensemble_0_cache.pkl"

frob_cost = GPNormalizedFrobeniusCostGenerator()
lang = OPENQASM2Language(gate_defs=[("gg", gg_gate_def)])


def calculate_probs_qp(H: np.ndarray, f: np.ndarray) -> np.ndarray[float]:
    M = H.shape[0]
    # Constraints, probabilities should sum to 1 and be between 0 and 1
    Aeq = np.ones((1, M))
    beq = np.array([1])
    lbound = np.zeros(M)
    ubound = np.ones(M)

    xk = solve_qp(H, f, A=Aeq, b=beq, lb=lbound, ub=ubound, solver='clarabel')                    

    return xk

def calculate_H_f(ensemble: np.ndarray, target: UnitaryMatrix) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate the Hessian and gradient for the Frank-Wolf algorithm.
    """
    M = ensemble.shape[0]
        
    tr_V_Us = np.einsum("mij,ij->m", ensemble, target.conj(), optimize=True)
    tr_Us = np.einsum("aij,bij->ab", ensemble.conj(), ensemble, optimize=True)

    f = -2 * np.real(tr_V_Us)
    H = 2 * np.real(tr_Us)

    # Make pos definite
    evs = np.linalg.eigvals(H)
    isposdef = np.all(evs > 0)
    trials = 0
    while not isposdef and trials < 20:
        H += 1e-10 * np.eye(M)
        print(f"Perturbing a little to make pos def try #: {trials}")
        evs = np.linalg.eigvals(H)
        isposdef = np.all(evs > 0)
        trials += 1

    if not isposdef:
        print('H not positive definite by a lot! Returning uniform dist')
        return None, None
    
    return H, f

def calculate_probs_frank_wolf(H: np.ndarray, f: np.ndarray,
                               initial_xk: np.ndarray = None) -> np.ndarray[float]:
    
    M = H.shape[0]
    if initial_xk is None:
        xk = np.ones(M) / M
    else:
        xk = initial_xk
    nSteps = 20
    # temp = np.zeros((4, nSteps))

    # Constraints, probabilities should sum to 1 and be between 0 and 1
    Aeq = np.ones((1, M))
    beq = np.array([1])
    lbound = np.zeros(M)
    ubound = np.ones(M)

    for _ in range(nSteps):
        DelJxk = H @ xk + f

        # Solve: min_y DelJxk.T @ y  s.t. Aeq @ y = beq, lbound <= y <= ubound
        # Using scipy.linprog for efficiency
        y_var = cp.Variable(M)
        lp_obj = cp.Minimize(DelJxk @ y_var)
        lp_constraints = [Aeq @ y_var == beq, y_var >= lbound, y_var <= ubound]
        lp_prob = cp.Problem(lp_obj, lp_constraints)
        lp_prob.solve(solver=cp.CVXOPT) 

        y = y_var.value
        step = y - xk

        # Optimal step size (gamma_star)
        numerator = -step @ DelJxk
        denominator = step @ H @ step
        gamma_star = min(1.0, numerator / denominator) if denominator > 1e-12 else 1.0

        # Update
        xk = xk + gamma_star * step
        print("diff: ", np.linalg.norm(gamma_star * step), flush=True)
    return xk

def calculate_bias_var_covar(uns: list[UnitaryMatrix], probs: list[float],
                             target: UnitaryMatrix):
    '''
    Calculate the bias, var, and covariance error terms
    '''
    avg_un = np.average(uns, axis=0, weights=probs)
    bias_err = gp_frobenius_cost(avg_un, target)

    var_err = 0
    for p, un in zip(probs,uns):
        var_err += (p ** 2) * (gp_frobenius_cost(avg_un, un) ** 2)

    covar_err = 0
    for i, (pi, ui) in enumerate(zip(probs, uns)):
        for j, (pj, uj) in enumerate(zip(probs, uns)):
            if i == j:
                continue
            u1 = ui - avg_un
            u2 = uj - avg_un
            covar_err += (pi * pj) * np.trace(u1.conj().T @ u2)

    covar_err = 2 * np.real(covar_err)

    return bias_err, var_err, covar_err

def random_gg_circ(num_qubits: int, num_ggs: int) -> Circuit:
    """
    Generate a random circuit with a specified number of GridSynthGates.
    """
    circ = Circuit(num_qubits)
    for _ in range(num_ggs):
        angle = np.random.uniform(0, 2 * np.pi)
        epsilon = 16
        z_twirl = np.random.randint(0, 2)
        rand_qubit = np.random.randint(0, num_qubits)
        circ.append_gate(GridSynthGate(), (rand_qubit, ), [angle, epsilon, z_twirl])
        rand_qubit_2 = np.random.randint(0, num_qubits)
        if rand_qubit != rand_qubit_2:
            circ.append_gate(U3Gate(), (rand_qubit_2, ), np.random.random(3) * 2 * np.pi)
            circ.append_gate(U3Gate(), (rand_qubit, ), np.random.random(3) * 2 * np.pi)
            circ.append_gate(CNOTGate(), (rand_qubit, rand_qubit_2))
    return circ

class CreateProbEnsemble(BasePass):

    def __init__(self, success_threshold: float = 1e-5, limit: int = 10):
        """
        A pass that generates a probability ensemble for a given circuit.
        """
        self.success_threshold = success_threshold
        self.limit = limit


    async def generate_un_prob(gg_inds: list[int],
                               circ: Circuit, 
                               target: UnitaryMatrix,
                               gg_probs: list[list[float]],
                               gg_param_options: list[list[str]]) -> tuple[list[UnitaryMatrix], 
                                                                           list[float]]:
        """
        Generate a unitary and its probability for a given circuit.
        """
        uns = []
        probs = []
        # print(f"Generating unitaries for {len(gg_inds)} selections", flush=True)
        # print(f"GG Indices: {gg_inds}", flush=True)
        for i, selections in enumerate(gg_inds):
            gg_ind = 0
            new_circ = circ.copy()
            gg_prob = [gg_probs[j][selection] for j, selection in enumerate(selections)]
            gg_strs = [gg_param_options[j][selection] for j, selection in enumerate(selections)]
            total_prob = np.prod(gg_prob)
            # if total_prob < 1e-6:
            #     continue
            # print("Selections: ", selections, "Total Prob: ", total_prob, flush=True)
            for cycle, op in new_circ.operations_with_cycles():
                if isinstance(op.gate, GridSynthGate):
                    t_str= gg_strs[gg_ind]
                    t_circ = gridsynth_gates_to_cir(t_str)
                    pt = CircuitPoint(cycle, op.location[0])
                    new_circ.replace_with_circuit(pt, t_circ, as_circuit_gate=True)
                    # rz_cost = frob_cost.calc_cost(t_circ, RZGate().get_unitary([op.params[0]]))
                    # print(f"Replaced GG at ({cycle}, qubit {op.location[0]}) w/"
                    #       f"RZ Cost: {rz_cost}, Prob: {gg_prob[gg_ind]}", flush=True)
                    gg_ind += 1
                    if gg_ind >= len(selections):
                        # Only modify first `limit` GGs
                        break
            # full_cost = frob_cost.calc_cost(new_circ, target)
            # print(f"Full Cost for circ {i+1}/{len(gg_inds)}: {full_cost}, Total Prob: {total_prob}", flush=True)
            uns.append(get_corrected_un(new_circ.get_unitary(), target))
            probs.append(total_prob)

        return uns, probs


    async def run(self, circ: Circuit, data: PassData) -> tuple[list[Circuit], 
                                                        list[float]]:
        success_threshold = self.success_threshold
        target = data.target
        dist = frob_cost.calc_cost(circ, target)
        # For each U3 gate, calculate do a Hamiltonian perturbation
        num_ggs = circ.count(GridSynthGate())
        # print(f"Number of GridSynthGates: {num_ggs}", flush=True)
        # print(circ.gate_counts, flush=True)
        

        gg_param_options: list[list[str]] = []
        gg_probs: list[list[float]] = []

        orig_perturb_dist = success_threshold - dist
        perturb_dist = orig_perturb_dist / (num_ggs + 1)
        # Round log of dist to nearest int
        int_perturb_dist = ceil(-1 * np.log10(perturb_dist))
        # print("Number of Circs: ", num_circs, flush=True)
        # Only perturb the first `limit` GGs
        for op in circ.operations():
            if isinstance(op.gate, GridSynthGate):
                angle = op.params[0]
                _, gg_strs, probs = get_rz_perturbations(angle, int_perturb_dist)
                gg_strs += ["Z" + s + "Z" for s in gg_strs]

                # Show angle reduction
                rz_target = RZGate().get_unitary([angle])
                gg_circs = [gridsynth_gates_to_cir(s) for s in gg_strs]
                gg_uns = [c.get_unitary() for c in gg_circs]
                gg_uns = [get_corrected_un(u, rz_target) for u in gg_uns]
                gg_param_options.append(gg_strs)
                gg_probs.append(probs)
                if len(gg_probs) >= self.limit:
                    break

        # Chunk into 1000 groups
        all_gg_inds = list(product(range(4), repeat=len(gg_probs)))
        # print(f"Total number of selections: {len(all_gg_inds)}", flush=True)
        all_gg_inds = np.array_split(all_gg_inds, 128)
        # print(f"Total number of selections: {len(all_gg_inds)}", flush=True)

        all_results = await get_runtime().map(
            CreateProbEnsemble.generate_un_prob, 
            all_gg_inds, 
            circ=circ, 
            target=target, 
            gg_probs=gg_probs, 
            gg_param_options=gg_param_options
        )

        uns = list(chain.from_iterable([r[0] for r in all_results]))
        probs = list(chain.from_iterable([r[1] for r in all_results]))
        # return uns, probs
        data["uns"] = uns
        data["probs"] = probs

if __name__ == '__main__':

    compiler = Compiler(num_workers=128)

    csv_data = []
    for gg_num in range(4, 6):
        # Random Circ
        for _ in range(3):
            circ = random_gg_circ(4, gg_num)
            target = circ.get_unitary()
            num_cnots = circ.count(CNOTGate())
            num_u3s = circ.count(U3Gate())

            _, data = compiler.compile(circ, [
                SetTargetPass(target),
                CreateProbEnsemble(success_threshold=(10 ** (-tol)),
                                limit=10),], 
                request_data=True)

            uns = data['uns']
            probs = data['probs']
                
            # Normalize probabilities
            probs = np.array(probs)
            if np.sum(probs) > 0:
                probs /= np.sum(probs)

            # Pick 10000 random unitaries from the ensemble
            NUM_CIRCS_PER_PROB = 4096
            if len(uns) > NUM_CIRCS_PER_PROB:
                rand_un_inds = np.random.choice(len(uns), size=NUM_CIRCS_PER_PROB, 
                                                replace=False)
                uns = [uns[i] for i in rand_un_inds]
                dists = [dists[i] for i in rand_un_inds]
                probs = [probs[i] for i in rand_un_inds]

            dists = [gp_frobenius_cost(u, target) for u in uns]

            ensemble = np.array(uns)
            H, f = calculate_H_f(ensemble, target)

            # Method 1: Uniform Probabilities
            uniform_probs = np.ones(len(uns)) / len(uns)
            # Calculate the bias, var, and covar errors
            bias_err, var_err, covar_err = calculate_bias_var_covar(uns, uniform_probs, target)
            avg_dist = np.mean(dists)
            gamma = bias_err / (avg_dist ** 2) 

            # Method 2: GG Probabilities
            bias_err_gg, var_err_gg, covar_err_gg = calculate_bias_var_covar(uns, probs, target)
            avg_dist_gg = np.average(dists, weights=probs)
            gamma_gg = bias_err_gg / (avg_dist_gg ** 2)

            # Calculate Probs with QP
            ensemble = np.array(uns)
            qp_probs = calculate_probs_qp(H, f)

            bias_err_qp, var_err_qp, covar_err_qp = calculate_bias_var_covar(uns, qp_probs, target)
            avg_dist_qp = np.average(dists, weights=qp_probs)
            gamma_qp = bias_err_qp / (avg_dist_qp ** 2)

            #Frank-Wolf algorithm
            fw_probs = calculate_probs_frank_wolf(H, f)
            bias_err_fw, var_err_fw, covar_err_fw = calculate_bias_var_covar(uns, fw_probs, target)
            avg_dist_fw = np.average(dists, weights=fw_probs)
            gamma_fw = bias_err_fw / (avg_dist_fw ** 2)
            
            # Frank-Wolf with seeded probabilities
            fws_probs = calculate_probs_frank_wolf(H, f, initial_xk=probs)
            bias_err_fws, var_err_fws, covar_err_fws = calculate_bias_var_covar(uns, fws_probs, target) 
            avg_dist_fws = np.average(dists, weights=fws_probs)
            gamma_fws = bias_err_fws / (avg_dist_fws ** 2)

            row = [avg_dist, len(uns), bias_err, var_err + covar_err, gamma,
                   bias_err_gg, var_err_gg + covar_err_gg, gamma_gg,
                   bias_err_qp, var_err_qp + covar_err_qp, gamma_qp,
                   bias_err_fw, var_err_fw + covar_err_fw, gamma_fw,
                   bias_err_fws, var_err_fws + covar_err_fws, gamma_fws]
            csv_data.append(row)

    compiler.close()
    # Write data to csv
    headers = ["Epsilon", "Num Circs", 
               "Bias Err (Uniform)", "Var Err + Covar Err (Uniform)", "Scaling Factor (Uniform)",
               "Bias Err GG", "Var Err + Covar Err GG", "Scaling Factor GG",
               "Bias Err QP", "Var Err + Covar Err QP", "Scaling Factor QP", 
               "Bias Err FW", "Var Err + Covar Err FW", "Scaling Factor FW",
               "Bias Err FW Seeded", "Var Err + Covar Err FW Seeded", "Scaling Factor FW Seeded"
               ]
    csv_file_name = 'gg_qp_all_results_rand_circs.csv'
    with open(csv_file_name, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(csv_data)
        print("Data written to ", csv_file_name, flush=True)