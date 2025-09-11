
import os
from typing import Generator
import numpy as np
from bqskit.runtime import get_runtime
from bqskit.ir.circuit import Circuit, CircuitPoint, CircuitLocationLike
from bqskit.ir.gates import ConstantUnitaryGate
from bqskit.qis import UnitaryMatrix
from .common import create_jiggled_unitaries
from .gg import gridsynth_gates_to_cir, GridSynthGate


def get_next_rho(params: np.array, circuit: Circuit,
                 rho_in: np.ndarray, qubits: CircuitLocationLike,
                 cache: dict, cliff_t: bool) -> np.ndarray:
    ''' Return the density matrix for a single circuit with a single parameter option. '''
    out_circ = circuit.copy()
    out_circ.set_params(params)
    if cliff_t:
        # Lower all GridSynthGates to corresponding Cliff T circs
        for cycle, op in circuit.operations_with_cycles():
            if isinstance(op.gate, GridSynthGate):
                pt = CircuitPoint(cycle, op.location[0])
                cache_ind = (op.params[0], int(op.params[1]))
                t_str = cache[cache_ind]
                if op.params[2] == 1:
                    t_str = "Z" + t_str + "Z"
                clifft_un = gridsynth_gates_to_cir(t_str).get_unitary()

                out_circ.replace_gate(pt, 
                                        ConstantUnitaryGate(clifft_un), 
                                        location=op.location)

    return get_single_rho(out_circ.get_unitary(), rho_in, qubits)

def get_file_names(large_checkpoint_dir, 
                   small_block_num: str,
                   no_qp: bool = False) -> tuple[str, str, str, str, str]:
    small_checkpoint_dir = os.path.join(large_checkpoint_dir, f"block_{small_block_num}")

    if no_qp:
        ensemble_file = os.path.join(small_checkpoint_dir, "ensemble_final_fw_no_qp.qasms")
        jiggle_file = os.path.join(small_checkpoint_dir, "ensemble_final_jiggle_fw_no_qp.npy")
        cache_file = os.path.join(small_checkpoint_dir, "ensemble_final_cache_fw_no_qp.pkl")
        csv_file = os.path.join(large_checkpoint_dir, f"block_{small_block_num}_fw_no_qp.csv")
        return ensemble_file, jiggle_file, cache_file, final_probs_file, csv_file

    # Try outputs of newest passes
    final_probs_file = os.path.join(small_checkpoint_dir, "ensemble_final_probs_fw.npy")
    if os.path.exists(final_probs_file):
        ensemble_file = os.path.join(small_checkpoint_dir, "ensemble_final_fw.qasms")
        jiggle_file = os.path.join(small_checkpoint_dir, "ensemble_final_jiggle_fw.npy")
        cache_file = os.path.join(small_checkpoint_dir, "ensemble_final_cache_fw.pkl")
        csv_file = os.path.join(large_checkpoint_dir, f"block_{small_block_num}_fw.csv")
        return ensemble_file, jiggle_file, cache_file, final_probs_file, csv_file
    
    # Otherwise, we do not have the newest set of files, so return the old ones
    csv_file = os.path.join(large_checkpoint_dir,
                             f"block_{small_block_num}_fw.csv")
    if not os.path.exists(csv_file):
        csv_file = os.path.join(large_checkpoint_dir, 
                                 f"block_{small_block_num}.csv")
    ensemble_file = os.path.join(small_checkpoint_dir, "ensemble_0__fw.qasms")
    jiggle_file = os.path.join(small_checkpoint_dir, "ensemble_0_jiggles__fw.npy")
    probs_file = os.path.join(small_checkpoint_dir, "ensemble_0_probs__fw.npy")
    cache_file = os.path.join(small_checkpoint_dir, "ensemble_0_cache__fw.pkl")
    return ensemble_file, jiggle_file, cache_file, probs_file, csv_file

def get_single_rho(u: UnitaryMatrix,
                   rho_in: np.ndarray, 
                   qubits: CircuitLocationLike) -> np.ndarray:
    '''
    Applies a single circuit unitary to the input density matrix rho_in at
    the specified qubits, returning the output density matrix.
    '''
    k = len(qubits)
    n = int(np.log2(rho_in.shape[0]))
    assert u.shape == (2**k, 2**k), f"Unitary shape {u.shape} does not match qubit count {k}"

    # Apply the unitary to the specified qubits
    rho_t = rho_in.reshape([2]*2*n) # shape = (2, 2, ..., 2) with 2n 2's

    ket_axes = list(range(n))
    target_ket = list(qubits)
    other_ket = [i for i in ket_axes if i not in target_ket]
    perm = target_ket + other_ket + [n + t for t in target_ket] + [n + i for i in other_ket]
    rho_perm = np.transpose(rho_t, axes=perm)
    # 3) reshape into blocks: (dT, dO, dT, dO) where dT = 2**k, dO = 2**(n-k)
    dT = 2**k
    dO = 2**(n - k)
    rho_block = rho_perm.reshape((dT, dO, dT, dO))
    # 4) apply U on the first (ket) index and U^\dagger on the first (bra) index using tensordot
    # temp = U (a,b) contracted with rho_block (b, A, d, C) over b -> temp (a, A, d, C)
    temp = np.tensordot(u, rho_block, axes=([1], [0]))
    # now contract temp (a, A, d, C) with U.conj().T (d, c) over d -> result (a, A, C, c)
    result = np.tensordot(temp, u.conj().T, axes=([2], [0]))  # shape (a, A, C, c)
    # transpose axes back to (a, A, c, C)
    result = np.transpose(result, axes=(0,1,3,2))
    # 5) reshape back and invert permutation
    rho_perm_back = result.reshape([2]*n + [2]*n)
    # compute inverse permutation
    inv_perm = np.argsort(perm)
    rho_out_t = np.transpose(rho_perm_back, axes=inv_perm)
    rho_out = rho_out_t.reshape((2**n, 2**n))
    return rho_out


class EnsembleSampler:
    def __init__(self, all_circ_params: list[tuple[Circuit, np.ndarray, np.ndarray, dict]],
                 cliff_t: bool = False):
        self.all_probs = [probs.flatten() for _, _, probs, _ in all_circ_params]
        self.circ_probs = np.array([np.sum(p) for p in self.all_probs])
        self.circ_probs /= np.sum(self.circ_probs)
        self.params = [params for _, params, _, _ in all_circ_params]
        self.circs = [circ for circ, _, _, _ in all_circ_params]
        self.caches = [cache for _, _, _, cache in all_circ_params]
        self.cliff_t = cliff_t

    def __call__(self) -> np.ndarray[np.complex128] | list[tuple[UnitaryMatrix, float]]:
        return create_jiggled_unitaries(self.circ_params, self.target, self.add_cost)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        # Create a random circuit based on the probabilities
        rand_circ_idx = np.random.choice(len(self.circ_probs), p=self.circ_probs)
        circ = self.circs[rand_circ_idx]
        param_options = self.params[rand_circ_idx]
        param_probs = self.all_probs[rand_circ_idx]
        rand_param_idx = np.random.choice(len(param_options), p=param_probs)
        rand_param = param_options[rand_param_idx]
        cache = self.caches[rand_circ_idx]
        out_circ = circ.copy()
        out_circ.set_params(rand_param)
        if self.cliff_t:
            # Lower all GridSynthGates to corresponding Cliff T circs
            for cycle, op in out_circ.operations_with_cycles():
                if isinstance(op.gate, GridSynthGate):
                    pt = CircuitPoint(cycle, op.location)
                    cache_ind = (op.params[0], int(op.params[1]))
                    t_str = cache[cache_ind]
                    if op.params[2] == 1:
                        t_str = "Z" + t_str + "Z"
                    clifft_circ = gridsynth_gates_to_cir(t_str)
                    out_circ.replace_with_circuit(pt, clifft_circ, as_circuit_gate=True)
            out_circ.unfold_all()
        return out_circ



class OrderedEnsembleSampler:
    def __init__(self, all_circ_params: list[tuple[Circuit, np.ndarray, np.ndarray, dict]],
                 cliff_t: bool = False):
        self.all_probs = [probs.flatten() for _, _, probs, _ in all_circ_params]
        self.params = [params for _, params, _, _ in all_circ_params]
        self.circs = [circ for circ, _, _, _ in all_circ_params]
        self.caches = [cache for _, _, _, cache in all_circ_params]
        self.cliff_t = cliff_t
        self.circ_ind = 0
        self.param_ind = 0
    
    def __iter__(self):
        return self

    def get_circ(self, param_ind: int, circ_ind: int) -> tuple[UnitaryMatrix, float]:
        circ = self.circs[circ_ind]
        param_options = self.params[circ_ind]
        param = param_options[param_ind]
        prob = self.all_probs[circ_ind][param_ind]
        cache = self.caches[circ_ind]
        out_circ = circ.copy()
        out_circ.set_params(param)
        if self.cliff_t:
            # Lower all GridSynthGates to corresponding Cliff T circs
            for cycle, op in out_circ.operations_with_cycles():
                if isinstance(op.gate, GridSynthGate):
                    pt = CircuitPoint(cycle, op.location[0])
                    cache_ind = (op.params[0], int(op.params[1]))
                    t_str = cache[cache_ind]
                    if op.params[2] == 1:
                        t_str = "Z" + t_str + "Z"
                    clifft_un = gridsynth_gates_to_cir(t_str).get_unitary()

                    out_circ.replace_gate(pt, 
                                          ConstantUnitaryGate(clifft_un), 
                                          location=op.location)

        return out_circ.get_unitary(), prob
    
    def __next__(self):
        un, prob = self.get_circ(self.param_ind, self.circ_ind)

        param_options = self.params[self.circ_ind]

        # Update indices
        self.param_ind += 1
        if self.param_ind >= len(param_options):
            self.param_ind = 0
            self.circ_ind += 1
            if self.circ_ind >= len(self.circs):
                raise StopIteration

        return un, prob
    
    async def get_next_batch_rhos(self, 
                                  rho_in: np.ndarray,
                                  qubits: CircuitLocationLike) -> np.ndarray | None:
        ''' Return all unitaries for a single circuit with all parameter options. '''
        if self.circ_ind >= len(self.circs):
            return None

        # Split params into 6 batches to avoid memory issues
        param_batches = np.array_split(self.params[self.circ_ind], 
                                       16, axis=0)
        probs_batches = np.array_split(self.all_probs[self.circ_ind], 
                                       16, axis=0)

        rho_out = np.zeros_like(rho_in, dtype=np.complex128)

        for param_batch, probs_batch in zip(param_batches, probs_batches):
            rhos = await get_runtime().map(get_next_rho,
                                            param_batch,
                                            circuit=self.circs[self.circ_ind],
                                            rho_in=rho_in,
                                            qubits=qubits,
                                            cache=self.caches[self.circ_ind],
                                            cliff_t=self.cliff_t)
            rho_out += np.tensordot(probs_batch, rhos, axes=([0], [0]))
        
        # Update circ index
        self.circ_ind += 1
        self.param_ind = 0

        return rho_out

    def reset(self) -> None:
        self.circ_ind = 0
        self.param_ind = 0

    async def output_rho(self, rho_in: np.ndarray, 
                   qubits: CircuitLocationLike) -> np.ndarray:
        rho_out = np.zeros_like(rho_in, dtype=np.complex128)
        done = False
        while not done:
            batch_rho = await self.get_next_batch_rhos(
                rho_in=rho_in,
                qubits=qubits
            )

            if batch_rho is None:
                done = True
                break
            
            rho_out += batch_rho

        # Assert that the output is a valid density matrix
        assert np.isclose(np.trace(rho_out), 1.0)
        return rho_out