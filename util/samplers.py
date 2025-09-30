import numpy as np
from bqskit.ir.circuit import Circuit, CircuitPoint, CircuitLocationLike
from bqskit.ir.gates import ConstantUnitaryGate
from bqskit.qis import UnitaryMatrix
from .distance import get_density_matrix
from .common import create_jiggled_unitaries, get_corrected_un
from .gg import gridsynth_gates_to_cir, GridSynthGate


def get_superop(params: np.array, circ: Circuit, 
                cache: dict, cliff_t: bool,
                target: UnitaryMatrix) -> np.ndarray:
    ''' Return the density matrix for a single circuit with a single parameter option. '''
    out_circ = circ.copy()
    out_circ.set_params(params)
    if cliff_t:
        # Lower all GridSynthGates to corresponding Cliff T circs
        for cycle, op in circ.operations_with_cycles():
            if isinstance(op.gate, GridSynthGate):
                pt = CircuitPoint(cycle, op.location[0])
                cache_ind = (op.params[0], int(op.params[1]))
                if cache_ind not in cache:
                    assert np.isclose(op.params[0], 0.0)
                    clifft_circ = Circuit(1)
                else:
                    t_str = cache[cache_ind]
                    if op.params[2] == 1:
                        t_str = "Z" + t_str + "Z"
                    clifft_circ = gridsynth_gates_to_cir(t_str)
                    
                clifft_un = clifft_circ.get_unitary()

                out_circ.replace_gate(pt, 
                                        ConstantUnitaryGate(clifft_un), 
                                        location=op.location)

    u = get_corrected_un(out_circ.get_unitary(), target)
    return np.kron(u.conj(), u), u

def reshape_rho(rho_in: np.ndarray) -> np.ndarray:
    num_qubits = int(np.log2(rho_in.shape[0]))
    assert rho_in.shape == (2**num_qubits, 2**num_qubits)
    # Reshape rho from matrix to tensor
    rho_tensor = rho_in.reshape([2]*2*num_qubits)
    return rho_tensor

def permute(rho_tensor: np.ndarray, 
            gate_loc: np.ndarray[int], 
            num_qubits: int) -> tuple[np.ndarray, list[int]]:
    # Move the qubits to be acted on
    target_ket = list(gate_loc)
    other_ket = [i for i in range(num_qubits) if i not in gate_loc]
    target_bra = [i + num_qubits for i in target_ket]
    other_bra = [i + num_qubits for i in other_ket]
    perm = target_ket + other_ket + target_bra + other_bra

    # Permute
    rho_perm = np.transpose(rho_tensor, axes=perm)
    return rho_perm, perm

def apply_small_superoperator(rho_perm: np.ndarray,
                              superop: np.ndarray,
                              gate_size: int,
                              num_qubits: int) -> np.ndarray:
    # Reshape into blocks
    dT = 2 ** gate_size
    dO = 2 ** (num_qubits - gate_size)
    rho_block = rho_perm.reshape((dT, dO, dT, dO)).transpose(0,2,1,3)

    # Reshape to matrix for superoperator application
    rho_targets = rho_block.reshape(dT, dT, dO*dO, order='F')  # shape (ketT, braT, rest)
    rho_targets = rho_targets.reshape(dT*dT, dO*dO, order='F')

    # Apply superoperator
    rho_targets = superop @ rho_targets

    # Reshape back to tensor that's same shape
    rho_targets = rho_targets.reshape(dT, dT, dO*dO, order='F')
    rho_targets = rho_targets.reshape(dT, dT, dO, dO, order='F')
    rho_targets = rho_targets.transpose(0,2,1,3)

    return rho_targets

def undo_perm(result: np.ndarray, 
              perm: list[int],
              num_qubits: int) -> np.ndarray:
    # Undo the reshape and permute
    rho_perm_back = result.reshape([2]*2*num_qubits)
    inv_perm = np.argsort(perm)
    rho_out_t = np.transpose(rho_perm_back, axes=inv_perm)
    rho_out_reshape = rho_out_t.reshape((2**num_qubits, 2**num_qubits))
    return rho_out_reshape

def apply_superoperator(rho_in: np.ndarray,
                        superop: np.ndarray,
                        num_qubits: int,
                        gate_loc: np.ndarray[int]) -> np.ndarray:
    '''
    Applies a superoperator to the input density matrix rho_in,
    returning the output density matrix.
    '''
    rho_tensor = reshape_rho(rho_in)
    rho_perm, perm = permute(rho_tensor, gate_loc, num_qubits)
    rho_block = apply_small_superoperator(rho_perm, superop, 
                                          len(gate_loc), num_qubits)
    rho_out_reshape = undo_perm(rho_block, perm, num_qubits)
    return rho_out_reshape

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
        # Clip very small probs to avoid numerical issues
        self.all_probs = np.array([np.clip(p, 1e-10, 1.0) for p in self.all_probs])
        # Normalize total probs
        self.all_probs /= np.sum(self.all_probs)

        self.circ_probs = np.array([np.sum(p) for p in self.all_probs])
        self.circ_probs /= np.sum(self.circ_probs)
        # Normalize each set of probs
        self.all_probs = [p / np.sum(p) for p in self.all_probs]
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

                    if cache_ind not in cache:
                        assert np.isclose(op.params[0], 0.0)
                        clifft_circ = Circuit(1)
                    else:
                        t_str = cache[cache_ind]
                        if op.params[2] == 1:
                            t_str = "Z" + t_str + "Z"
                        
                        clifft_circ = gridsynth_gates_to_cir(t_str)
                    out_circ.replace_with_circuit(pt, clifft_circ, as_circuit_gate=True)
            out_circ.unfold_all()
        return out_circ


class EnsembleUnitarySampler:
    def __init__(self, all_circ_params: list[tuple[Circuit, np.ndarray, np.ndarray, dict]],
                 cliff_t: bool = False):
        self.all_probs = [probs.flatten() for _, _, probs, _ in all_circ_params]
        self.params = [params for _, params, _, _ in all_circ_params]
        self.circs = [circ for circ, _, _, _ in all_circ_params]
        self.caches = [cache for _, _, _, cache in all_circ_params]
        self.cliff_t = cliff_t
        self.circ_ind = 0
        self.param_ind = 0

    def __call__(self) -> np.ndarray[np.complex128] | list[tuple[UnitaryMatrix, float]]:
        return create_jiggled_unitaries(self.circ_params, self.target, self.add_cost)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        # Create a random circuit based on the probabilities
        if self.circ_ind >= len(self.circs):
            raise StopIteration
        circ = self.circs[self.circ_ind]
        param_options = self.params[self.circ_ind]
        params = param_options[self.param_ind]
        probs = self.all_probs[self.circ_ind][self.param_ind]

        if self.cliff_t:
            cache = self.caches[self.circ_ind]
            out_circ = circ.copy()
            out_circ.set_params(params)
            # Lower all GridSynthGates to corresponding Cliff T circs
            for cycle, op in out_circ.operations_with_cycles():
                if isinstance(op.gate, GridSynthGate):
                    pt = CircuitPoint(cycle, op.location[0])
                    cache_ind = (op.params[0], int(op.params[1]))
                    if cache_ind not in cache:
                        assert np.isclose(op.params[0], 0.0)
                        clifft_circ = Circuit(1)
                    else:
                        t_str = cache[cache_ind]
                        if op.params[2] == 1:
                            t_str = "Z" + t_str + "Z"
                        clifft_circ = gridsynth_gates_to_cir(t_str)
                    out_circ.replace_with_circuit(pt, clifft_circ, as_circuit_gate=True)
            un = out_circ.get_unitary()
        else:
            un = circ.get_unitary(params)

        # Update Indices
        self.param_ind += 1
        if self.param_ind >= len(param_options):
            self.param_ind = 0
            self.circ_ind += 1

        return un, probs
    
    def get_rho_out(self, sv: np.ndarray) -> np.ndarray:
        # ONLY WORKS FOR NISQ CIRCUITS
        rho = get_density_matrix(sv)
        empty_rho = np.zeros_like(rho)
        print("Getting Rho Out", self.cliff_t, flush=True)
        for i, circ in enumerate(self.circs):
            cache = self.caches[i]
            if self.cliff_t:
                assert cache is not None
            for j, param in enumerate(self.params[i]):
                if self.cliff_t:
                    out_circ = circ.copy()
                    out_circ.set_params(param)
                    # Lower all GridSynthGates to corresponding Cliff T circs
                    for cycle, op in out_circ.operations_with_cycles():
                        if isinstance(op.gate, GridSynthGate):
                            pt = CircuitPoint(cycle, op.location[0])
                            cache_ind = (op.params[0], int(op.params[1]))
                            if cache_ind not in cache:
                                assert np.isclose(op.params[0], 0.0)
                                clifft_circ = Circuit(1)
                            else:
                                t_str = cache[cache_ind]
                                if op.params[2] == 1:
                                    t_str = "Z" + t_str + "Z"
                                clifft_circ = gridsynth_gates_to_cir(t_str)
                            out_circ.replace_with_circuit(pt, clifft_circ, as_circuit_gate=True)
                    out_circ.unfold_all()
                    # print(out_circ.gate_counts)
                    un = out_circ.get_unitary()
                else:
                    un = circ.get_unitary(param)
                p = self.all_probs[i][j]
                empty_rho += p * (un @ rho @ un.conj().T)
        return empty_rho