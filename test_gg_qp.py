# Now let's see if GenerateProbabilityPass can work on full ensemble
from bqskit.runtime import get_runtime

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData
from itertools import chain, product
from util.generate_probs_pass import GenerateProbabilityPass
import numpy as np
from bqskit.ir.gates import CNOTGate, U3Gate
from bqskit.qis import UnitaryMatrix
from math import ceil
from util.gg import get_rz_perturbations, gg_gate_def, GridSynthGate, gridsynth_gates_to_cir
from bqskit.ir.gates import RZGate
from bqskit.passes import ScanPartitioner, SetTargetPass
from bqskit.compiler import Compiler
from bqskit.ir.circuit import Circuit, CircuitGate, CircuitPoint
from util.distance import frobenius_cost, normalized_gp_frob_cost
from util.common import load_block, load_ensemble
from util.distance import get_corrected_un
import csv
from bqskit.ir.lang.qasm2 import OPENQASM2Language
from bqskit.ir.opt.cost.functions import GPNormalizedFrobeniusCostGenerator, GPNormalizedFrobeniusCostGenerator

circ_name = "qpe10"
block_name = "2"
tol = 3.0
ind = 0
small_block = "00"


qasms_file = f"/pscratch/sd/j/jkalloor/bqskit/small_block_checkpoints_final_paper_4_clifft_tket/{circ_name}_{block_name}_{tol}/block_{small_block}/ensemble_0_.qasms"
jiggle_file = f"small_block_checkpoints_final_paper_4_clifft_tket/{circ_name}_{block_name}_{tol}/block_{small_block}/ensemble_0_jiggles_.npy"
cache_file = f"/pscratch/sd/j/jkalloor/bqskit/small_block_checkpoints_final_paper_4_clifft_tket/{circ_name}_{block_name}_{tol}/block_{small_block}/ensemble_0_cache.pkl"

frob_cost = GPNormalizedFrobeniusCostGenerator()
lang = OPENQASM2Language(gate_defs=[("gg", gg_gate_def)])

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
        new_circ = circ.copy()
        # print(f"Generating unitaries for {len(gg_inds)} selections", flush=True)
        # print(f"GG Indices: {gg_inds}", flush=True)
        for i, selections in enumerate(gg_inds):
            gg_ind = 0
            gg_prob = [gg_probs[j][selection] for j, selection in enumerate(selections)]
            gg_strs = [gg_param_options[j][selection] for j, selection in enumerate(selections)]
            total_prob = np.prod(gg_prob)
            # if total_prob < 1e-6:
            #     continue
            for cycle, op in new_circ.operations_with_cycles():
                if isinstance(op.gate, GridSynthGate):
                    t_str= gg_strs[gg_ind]
                    t_circ = gridsynth_gates_to_cir(t_str)
                    pt = CircuitPoint(cycle, op.location[0])
                    new_circ.replace_with_circuit(pt, t_circ, as_circuit_gate=True)
                    gg_ind += 1
                    if gg_ind >= len(selections):
                        # Only modify first `limit` GGs
                        break
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
                avg_dist = np.mean([frob_cost(c, rz_target) for c in gg_circs])
                avg_un = np.average(gg_uns, axis=0, weights=probs)
                avg_un_dist = normalized_gp_frob_cost(avg_un, rz_target)
                gamma = avg_un_dist / (avg_dist ** 2)
                print(f"Angle: {angle}, Avg Dist: {avg_dist}, "
                      f"Avg Unitary Dist: {avg_un_dist}, "
                      f"Gamma: {gamma}", flush=True)
                gg_param_options.append(gg_strs)
                gg_probs.append(probs)
                if len(gg_probs) > self.limit:
                    break

        # Chunk into 1000 groups
        all_gg_inds = list(product(range(4), repeat=len(gg_probs)))
        # print(f"Total number of selections: {len(all_gg_inds)}", flush=True)
        all_gg_inds = np.array_split(all_gg_inds, 256)
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

    orig_circ = load_block("qpe10", "2", "_tket")
    orig_circ = Circuit.from_file(orig_circ)
    # print(orig_circ.gate_counts, flush=True)

    compiler = Compiler(num_workers=256)
    part_circ = compiler.compile(orig_circ, [ScanPartitioner(4)])
    block_target_circs = {}
    num_digits = len(str(part_circ.num_operations))
    for i, op in enumerate(part_circ.operations()):
        assert isinstance(op.gate, CircuitGate)
        label = str(i).zfill(num_digits)
        block_target_circs[label] = op.gate._circuit
    target_circ: Circuit = block_target_circs[small_block]
    target = target_circ.get_unitary()

    ensemble = load_ensemble(qasms_file)
    circ = ensemble[int(small_block)]
    csv_data = []
    for gg_num in range(3, 8):
        # 4 random circuits for each number of GridSynthGates
        # for _ in range(4):
        # num_ggs = circ.count(GridSynthGate())
        num_cnots = circ.count(CNOTGate())
        num_u3s = circ.count(U3Gate())

        dist = frob_cost.calc_cost(circ, target)

        _, data = compiler.compile(circ, [
            SetTargetPass(target),
            CreateProbEnsemble(success_threshold=(10 ** (-tol)),
                               limit=gg_num),], 
            request_data=True)

        uns = data['uns']
        probs = data['probs']
            
        # Normalize probabilities
        probs = np.array(probs)
        if np.sum(probs) > 0:
            probs /= np.sum(probs)

        dists = [normalized_gp_frob_cost(u, target) for u in uns]
        avg_dist_1 = np.mean(dists)
        # print("Avg Distance: ", avg_dist)

        avg_un = np.average(uns, axis=0, weights=probs)

        avg_un_dist_1 = normalized_gp_frob_cost(avg_un, target)
        # print("Avg Unitary Distance: ", avg_un_dist)

        gamma = avg_un_dist_1 / (avg_dist_1 ** 2)
        # print("Scaling factor gamma: ", gamma, flush=True) 

        # Pick 10000 random unitaries from the ensemble
        NUM_CIRCS_PER_PROB = 4000
        if len(uns) > NUM_CIRCS_PER_PROB:
            rand_un_inds = np.random.choice(len(uns), size=NUM_CIRCS_PER_PROB, 
                                            replace=False)
            uns = [uns[i] for i in rand_un_inds]
            dists = [dists[i] for i in rand_un_inds]
        
        avg_dist_qp = np.mean(dists)

        ensemble = np.array(uns)

        qp_probs = GenerateProbabilityPass.calculate_probs(ensemble, target=target)

        avg_un_qp = np.average(ensemble, axis=0, weights=qp_probs)
        avg_un_qp_dist = normalized_gp_frob_cost(avg_un_qp, target)
        # print("Avg Unitary QP Distance: ", avg_un_qp_dist)

        gamma_qp = avg_un_qp_dist / (avg_dist_qp ** 2)
        # print("Scaling factor gamma QP: ", gamma_qp)
        row = [gg_num, num_cnots, num_u3s, avg_dist_1, avg_un_dist_1, gamma, 
            avg_dist_qp, avg_un_qp_dist, gamma_qp]
        
        print(row, flush=True)

        csv_data.append(row)

    compiler.close()
    # Write data to csv
    headers = ["Num GGs", "Num CNOTs", "Num U3s", "Avg. Dist",
               "Avg. Unitary Dist", "Gamma", "Avg. Dist QP",
               "Avg. Unitary Dist QP", "Gamma QP"]
    with open('gg_qp_results_qpe.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(csv_data)
        print("Data written to gg_qp_results.csv", flush=True)