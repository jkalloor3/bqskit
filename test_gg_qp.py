# Now let's see if GenerateProbabilityPass can work on full ensemble
from bqskit.runtime import get_runtime

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from itertools import chain, product
from util.generate_probs_pass import GenerateProbabilityPass
from util.check_ensemble_quality import CheckEnsembleQualityPass
from util import JiggleEnsemblePass
import numpy as np
import os
# import time
from bqskit.ir.gates import CNOTGate, U3Gate
from bqskit.qis import UnitaryMatrix
from math import ceil
from util.gg import get_rz_perturbations, gg_gate_def, GridSynthGate, gridsynth_gates_to_cir
from bqskit.ir.gates import RZGate
from bqskit.passes import CheckpointRestartPass, ForEachBlockPass
from bqskit.compiler import Compiler
from bqskit.ir.circuit import Circuit, CircuitGate, CircuitPoint
from util.distance import frobenius_cost, normalized_gp_frob_cost, gp_frobenius_cost
from util.common import load_block, load_ensemble, load_jiggled_ensemble, create_jiggled_unitaries
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


class GenerateAllProbabilitiesPass(BasePass):
    """
    A pass that generates all probabilities for a given circuit.
    This pass is used to generate the probabilities for the ensemble.
    """

    def __init__(self, checkpoint_extra_str: str = ""):
        self.checkpoint_extra_str = checkpoint_extra_str

    async def run(self, circ: Circuit, data: PassData) -> None:
        print("Running Generate All Probabilities Pass", flush=True)
        checkpoint_dir = data["checkpoint_dir"]
        ensemble_file_name = os.path.join(checkpoint_dir, "ensemble_{ind}_{extra}.qasms")
        jiggle_file_name = os.path.join(checkpoint_dir, "ensemble_{ind}_jiggles_{extra}.npy")
        probs_file_name = os.path.join(checkpoint_dir, "ensemble_{ind}_probs_{extra}.npy")
        cache_file_name = os.path.join(checkpoint_dir, "ensemble_{ind}_cache_{extra}.pkl")
        
        ens_file = ensemble_file_name.format(ind=0, extra=self.checkpoint_extra_str)
        jiggle_file = jiggle_file_name.format(ind=0, extra=self.checkpoint_extra_str)
        probs_file = probs_file_name.format(ind=0, extra=self.checkpoint_extra_str)
        cache_file = cache_file_name.format(ind=0, extra=self.checkpoint_extra_str)

        if not os.path.exists(ens_file):
            print(f"Ensemble file {ens_file} does not exist, skipping pass", flush=True)
            data["row_data"] = {}
            return
        
        circ_params = load_jiggled_ensemble(ens_file, jiggle_file,
                                            cache_file, probs_file)
        
        target = data.target

        print("Target shape: ", target.shape, flush=True)
        print("Circuit Qudits: ", circ.num_qudits, flush=True)
        
        orig_circs = load_ensemble(ens_file)
        
        orig_uns = []
        all_caches = [c for _, _, _, c in circ_params]
        for i , c in enumerate(orig_circs):
            if all_caches[i] is not None:
                w_cache = get_runtime().get_cache()
                w_cache.clear()
                w_cache.update(all_caches[i])
            orig_uns.append(get_corrected_un(c.get_unitary(), target))

        ensemble: list[list[UnitaryMatrix]] = await get_runtime().map(create_jiggled_unitaries, circ_params, 
                                            target=target, add_cost=False)
        
        # flat_ensemble = np.concatenate(ensemble)
        flat_ensemble = list(chain.from_iterable(ensemble))
        flat_ensemble = [UnitaryMatrix(u) for u in flat_ensemble]
        all_dists = []
        for i, sub_ens in enumerate(ensemble):
            all_dists.append([frobenius_cost(u, target) for u in sub_ens])

        flat_dists = np.concatenate(all_dists)

        # Initial probs from Gridsynth 
        all_probs = np.load(probs_file)

        # GG probs (uniform across all circs)
        outer_probs_vector = np.ones(len(orig_circs)) / len(orig_circs)
        gg_probs = [[p * qp_p for p in probs] for probs, qp_p in zip(all_probs, 
                                                                       outer_probs_vector)]
        
        # FW Seeded Algorithm
        orig_circs = load_ensemble(ens_file)
        
        orig_uns = []
        all_caches = [c for _, _, _, c in circ_params]
        for i , c in enumerate(orig_circs):
            if all_caches[i] is not None:
                w_cache = get_runtime().get_cache()
                w_cache.clear()
                w_cache.update(all_caches[i])
            orig_uns.append(get_corrected_un(c.get_unitary(), target))

        orig_ensemble = np.array(orig_uns)
        outer_probs_vector = GenerateProbabilityPass.calculate_probs(orig_ensemble,
                                                             target)
        fw_init_probs = [[p * qp_p for p in probs] for probs, qp_p in zip(all_probs, 
                                                                       outer_probs_vector)]
        
        if len(fw_init_probs) > 6000:
            # For each circ, pick most probable
            num_params_per_circ = ceil(6000 / len(orig_circs))

            new_inds = []
            for probs in gg_probs:
                # Sort indices by probs
                sorted_inds = np.argsort(probs)[::-1]
                new_inds.append(sorted_inds[:num_params_per_circ].tolist())

            # Now pick from ensemble and fw_init_probs those inds
            init_probs = []
            sub_ensemble = []
            sub_dists = []
            for i, inds in enumerate(new_inds):
                init_probs.append([fw_init_probs[i][j] for j in inds])
                sub_ensemble.append([ensemble[i][j] for j in inds])
                sub_dists.append([all_dists[i][j] for j in inds])

        else:
            sub_ensemble = ensemble
            init_probs = fw_init_probs
            sub_dists = all_dists


        # Pass probs to FW again
        init_probs = np.concatenate(init_probs)
        # sub_ensemble = np.concatenate(sub_ensemble)
        sub_ensemble: list[UnitaryMatrix] = list(chain.from_iterable(sub_ensemble))
        sub_ensemble = [UnitaryMatrix(u) for u in sub_ensemble]
        fw_seeded_probs = GenerateProbabilityPass.calculate_probs(
            ensemble=np.array(sub_ensemble), target=target, initial_probs=init_probs)
    
        
        # Now calculate average unitaries and error terms
        flat_gg_probs = np.concatenate(gg_probs)
        gg_bias_err, gg_var_err, gg_covar_err = calculate_bias_var_covar(flat_ensemble, flat_gg_probs, target)
        gg_avg_dist = np.average(flat_dists, weights=flat_gg_probs)
        gamma = gg_bias_err / (gg_avg_dist ** 2)  

        # FW Init Probs
        flat_fw_init_probs = np.concatenate(fw_init_probs)
        fw_bias_err, fw_var_err, fw_covar_err = calculate_bias_var_covar(
            flat_ensemble, flat_fw_init_probs, target)
        fw_avg_dist = np.average(flat_dists, weights=flat_fw_init_probs)
        fw_gamma = fw_bias_err / (fw_avg_dist ** 2)

        # FW Seeded Probs
        fw_seeded_bias_err, fw_seeded_var_err, fw_seeded_covar_err = calculate_bias_var_covar(
            sub_ensemble, fw_seeded_probs, target)

        # Calculate gamma
        fw_seeded_avg_dist = np.average(flat_dists, weights=fw_seeded_probs)
        fw_seeded_gamma = fw_seeded_bias_err / (fw_seeded_avg_dist ** 2)

        data["row_data"] = {
            "Num Original Circs": len(orig_circs),
            "Orig Ensemble Size": len(flat_ensemble),
            "Sub Ensemble Size": len(sub_ensemble),

            "Bias Err (GG)": gg_bias_err,
            "Var Err + Covar Err (GG)": gg_var_err + gg_covar_err,
            "Avg. Dist (GG)": gg_avg_dist,
            "Scaling Factor (GG)": gamma,

            "Bias Err (FW Init)": fw_bias_err,
            "Var Err + Covar Err (FW Init)": fw_var_err + fw_covar_err,
            "Avg. Dist (FW Init)": fw_avg_dist,
            "Scaling Factor (FW Init)": fw_gamma,

            "Bias Err (FW Seeded)": fw_seeded_bias_err,
            "Var Err + Covar Err (FW Seeded)": fw_seeded_var_err + fw_seeded_covar_err,
            "Avg. Dist (FW Seeded)": fw_seeded_avg_dist,
            "Scaling Factor (FW Seeded)": fw_seeded_gamma
        }

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

    compiler = Compiler(num_workers=256)
    base_checkpoint_dir_form = "small_block_checkpoints_final_paper_{block_size}_clifft_tket"

    all_id_data = {}

    SMALL_BLOCK_SIZE = 4

    # Random ensembles to test
    all_circ_data = [
        ("add17", "00", 4.0),
        # ("mult16", "05", 3.0),
        # ("mult16", "05", 2.0),
        # ("qpe10", "0", 2.0),
        # ("qpe10", "1", 4.0),
        # ("qpe10", "2", 3.0),
    ]
    all_row_data = []

    for circ_data in all_circ_data:
        circ_name, block_name, tol = circ_data
        circ = load_block(circ_name, block_name, extra="_tket")
        circ = Circuit.from_file(circ)

        base_checkpoint_dir = base_checkpoint_dir_form.format(block_size=SMALL_BLOCK_SIZE)
        checkpoint_dir = f"{base_checkpoint_dir}/{circ_name}_{block_name}_{tol}/"

        err_thresh = 10 ** (-1 * tol) / 2

        jiggle_pass = JiggleEnsemblePass(success_threshold=err_thresh, 
                            num_circs=2000, 
                            use_scan_sols=True,
                            use_ensemble=False,
                            use_calculated_error=False,
                            jiggle_skew=0,
                            count_t=True,
                            do_u3_perturbation=True,
                            flood_circ=False,
                            checkpoint_extra_str="_testing")

        id = compiler.submit(circ, [
            CheckpointRestartPass(checkpoint_dir, 
                        default_passes=[]),
            ForEachBlockPass(
                [
                jiggle_pass,
                GenerateProbabilityPass(
                    run_on_ensemble_0=True,
                    checkpoint_extra_str="_testing",
                ),
                CheckEnsembleQualityPass(
                    checkpoint_extra_str="_testing",
                )
                # GenerateAllProbabilitiesPass(
                #     checkpoint_extra_str="_testing"
                # )
                ])
            ], request_data=True)
        all_id_data[id] = circ_data

    for id, circ_data in all_id_data.items():
        _, all_data = compiler.result(id)
        circ_name, block_name, tol = circ_data
        block_data = all_data[ForEachBlockPass.key][0]
        for i, data in enumerate(block_data):
            row_data = {}
            row_data["circ_name"] = circ_name
            row_data["block_name"] = block_name
            row_data["tol"] = tol
            row_data["block_num"] = str(i)
            if len(data["row_data"]) > 0:
                row_data.update(data["row_data"])
                all_row_data.append(row_data)

    compiler.close()

    csv_file_name = 'gg_qp_ckpt_results.csv'
    headers = list(all_row_data[0].keys())
    all_row_data = [list(row[h] for h in headers) for row in all_row_data]
    with open(csv_file_name, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(all_row_data)
        print("Data written to ", csv_file_name, flush=True)