from qiskit_aer.noise import (NoiseModel, depolarizing_error, 
                              mixed_unitary_error)
from bqskit.ir.gates import CNOTGate, RXGate, IdentityGate, XGate, YGate, ZGate
import numpy as np
from bqskit.ir.circuit import Circuit


def over_rotated_cnot(eps: float) -> Circuit:
    '''
    Over-rotated CNOT gate.

    Return a list of unitary matrices and probabilities.

    We are assuming that we always over-rotate the CNOT gate by eps.
    Then, with probability eps, we also have Pauli X, Y, and Z errors
    '''
    # Over-rotate X (not as much as eps, but a fraction of it)
    effect = np.kron(IdentityGate().get_unitary(), RXGate().get_unitary([eps / 5]))

    # print("Effect: ", effect)
    # print("RX Gate: ", RXGate().get_unitary([eps]))

    probs = [1 - eps, eps/3, eps/3, eps/3]

    errors = []
    # pauli errors
    for i, q1_err in enumerate([IdentityGate(), XGate(), YGate(), ZGate()]):
        for j, q2_err in enumerate([IdentityGate(), XGate(), YGate(), ZGate()]):
            err = np.kron(q1_err.get_unitary(), q2_err.get_unitary())
            p1 = probs[i]
            p2 = probs[j]
            effect = effect @ err
            errors.append((effect, p1 * p2))

    # Assert probabilities sum to 1
    assert np.isclose(sum([p for _, p in errors]), 1.0)

    return errors


def get_nisq_noise_model(one_q_err: float = 1e-4, 
                         two_q_err: float = 1e-3) -> NoiseModel:
    '''
    Noise Model with overrotated CNOT and stochastic depolarizing noise.

    '''
    # Create an empty noise model
    noise_model = NoiseModel()

    # Add depolarizing error to all single qubit u1, u2, u3 gates
    one_q_error = depolarizing_error(one_q_err, 1)
    noise_model.add_all_qubit_quantum_error(one_q_error, ['u', 'u3'])

    coherent_utries = over_rotated_cnot(two_q_err)
    cnot_error = mixed_unitary_error(coherent_utries)
    noise_model.add_all_qubit_quantum_error(cnot_error, ['cx'])

    return noise_model

def get_clifft_noise_model(logical_err: float = 1e-10,
                           t_gate_err: float = 1e-7) -> NoiseModel:
    
    l_err_1 = depolarizing_error(logical_err, 1)
    l_err_2 = depolarizing_error(logical_err, 2)
    t_err = depolarizing_error(t_gate_err, 1)

    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(l_err_1, ['s', 'sdg', 'h'])
    noise_model.add_all_qubit_quantum_error(l_err_2, ['cx'])
    noise_model.add_all_qubit_quantum_error(t_err, ['t', 'tdg'])

    return noise_model