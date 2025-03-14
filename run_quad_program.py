import numpy as np
from bqskit.runtime import get_runtime
from qpsolvers import solve_qp, solve_ls
import matplotlib.pyplot as plt
from util import load_jiggled_ensemble, create_jiggled_unitaries
import time
# from scipy.linalg.blas import zgemm

from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import U3Gate
from bqskit.compiler import Compiler
from bqskit.passes import ForEachBlockPass, UpdateDataPass

from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData


def quad_program(ensemble: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Calculate the probabilities for the ensemble"""
    M = ensemble.shape[0]

    # ensemble is of size (20000, 256, 256) complex 128
    start = time.time()
    tr_V_Us = np.einsum("mij,ij->m", ensemble, target.conj(), optimize=True)
    dist_calc = time.time() - start
    print("Time to calculate dist", dist_calc, flush=True)
    start = time.time()
    tr_Us = np.einsum("aij,bij->ab", ensemble.conj(), ensemble, optimize=True)
    covar_calc = time.time() - start
    print(tr_Us.shape, flush=True)
    print("Time to calculate covar", covar_calc, flush=True)

    # tr_V_Us is of size (20000,)
    # tr_Us is of size (20000, 20000) floats 64

    # sample_inds = np.random.choice(M, size=5, replace=True)
    # print("Sample Dists", [tr_V_Us[i] for i in sample_inds], flush=True)
    # print("Sample Covars", [tr_Us[i][0] for i in sample_inds], flush=True)

    # print("Finished Calculating Ensemble Vectors", flush=True)
    # print(tr_V_Us.shape, tr_Us.shape, flush=True)
    start = time.time()
    # Create f and H matrices
    f = -2 * np.real(tr_V_Us)
    H = 2 * np.real(tr_Us)

    # Make pos definite
    isposdef = False
    trials = 0
    while not isposdef and trials < 20:
        try:
            R = np.linalg.cholesky(H)
            isposdef = True
        except np.linalg.LinAlgError:
            # Off by a little
            H += 1e-10 * np.eye(M)
            print(f"Perturbing by a little to make pos def trial num: {trials}")
            isposdef = False
            trials += 1

    if not isposdef:
        print('H not positive definite by a lot! Returning uniform dist')
        return [1 / len(ensemble) for _ in ensemble]
    
    total_time = time.time() - start
    print("Time to create f and H", total_time, flush=True)

    # Constraints, probabilities should sum to 1 and be between 0 and 1
    Aeq = np.ones((1, M))
    beq = np.array([1])
    lbound = np.zeros(M)
    ubound = np.ones(M)

    # Solve with LS since it is convex
    start = time.time()
    # s = -1 * np.linalg.inv(R) @ f
    # probabilities = solve_ls(R.T, s, None, None, Aeq, beq, lbound, ubound, solver='clarabel')
    probabilities = solve_qp(H, f, A=Aeq, b=beq, lb=lbound, ub=ubound, solver='clarabel')
    total_time = time.time() - start
    print("Time to solve", total_time, flush=True)

    # sample_probs = np.random.choice(probabilities, size=10, replace=True)
    # print("Sample Probs", sample_probs, flush=True)

    return probabilities


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    return np.sum(p * np.log(p / q))

def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)

class Temp(BasePass):
    def __init__(
        self,
        ensemble_file: str,
        jiggle_file: str,
    ) -> None:
        self.ensemble_file = ensemble_file
        self.jiggle_file = jiggle_file
     
    async def run(
            self, 
            circ : Circuit, 
            data: PassData
    ) -> None:
        circ_params = load_jiggled_ensemble(self.ensemble_file, 
                                            self.jiggle_file)
        # Load target
        target = circ.get_unitary()

        # Get all unitaries
        print("Calculating Unitaries", flush=True)
        start = time.time()
        ensemble = await get_runtime().map(create_jiggled_unitaries, circ_params, target=target, add_cost=False)
        ensemble = np.concatenate(ensemble, axis=0)
        # ensemble = np.array(ensemble, dtype=np.complex128)
        total_time = time.time() - start
        print("Total Time for Unitaries: ", total_time, flush=True)
        print("Ensemble Shape: ", ensemble.shape, flush=True)
        print("Ensemble Bytes (GB): ", ensemble.nbytes / 1024 / 1024 / 1024, flush=True)
        # print("Finished Calculating Unitaries", flush=True)

        # Now calculate the probabilities for 1000
        rand_un_inds = np.random.choice(ensemble.shape[0], size=10000, replace=False)
        rand_ensemble = ensemble[rand_un_inds]

        # Calculate the probabilities
        print("Calculating Probabilities", flush=True)
        start = time.time()
        p = quad_program(rand_ensemble, target=target)
        total_time = time.time() - start
        print("Total Time for Probabilities: ", total_time, flush=True)

        q = np.ones(len(rand_un_inds)) / len(rand_un_inds)
        # print JS Divergence between two distributions
        p = np.clip(p, 1e-12, 1)
        q = np.clip(q, 1e-12, 1)
        js_div= js_divergence(p, q)
        print("JS Divergence: ", js_div, flush=True)
        print(np.sum(p), np.sum(q), flush=True)
        print(np.max(p), np.min(p), np.std(p), flush=True)



if __name__ == '__main__':

    # Load ensemble
    ens_file = "block_checkpoints_final_paper_tket/qae11_1_0.5/ensemble_final.qasms"
    jiggle_file = "block_checkpoints_final_paper_tket/qae11_1_0.5/ensemble_final_jiggle.npy"
    
    # Load file
    circ = Circuit.from_file("good_blocks/qae11_1.qasm")

    print("Num Qubits: ", circ.num_qudits, flush=True)

    workflow = [
        Temp(ens_file, jiggle_file),
    ]

    compiler = Compiler(num_workers=256)
    compiler.compile(circ, workflow)
