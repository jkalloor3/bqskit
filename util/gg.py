"""This module implements the RZGate."""
from __future__ import annotations

import numpy as np

# from multiprocessing import shared_memory, Lock
from typing import TYPE_CHECKING
from cachetools import LRUCache

from bqskit.ir.circuit import Circuit, CircuitLocation
from bqskit.ir.gates import (IdentityGate, ZGate, SGate, SdgGate, 
                            TGate, HGate, TdgGate, XGate, RZGate, YGate)
    
from bqskit.ir.gates.qubitgate import QubitGate
from bqskit.qis.unitary.unitary import RealVector
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.cachedclass import CachedClass
from bqskit.utils.math import unitary_log_no_i, pauli_expansion
from pyLIQTR.gate_decomp.gate_approximation import approximate_rz_direct

from .fix_global_phase import fix_phase
from .distance import gp_frobenius_cost
from fractions import Fraction

MIN_EPSILON = 17

from bqskit.ir.lang.qasm2.visitor import GateDef

gate_defs = {'I': IdentityGate(1), 'Z': ZGate(), 'S': SGate(), 'Sd': SdgGate(), 
             'T': TGate(), 'Td': TdgGate(), 'H': HGate(), 'D': SdgGate(), 
             'X': XGate(), 'L': SdgGate(), 'Y': YGate()}

def gridsynth_gates_to_cir(gates: str):
    circ = Circuit(1)
    # Loop through string and add gates to circuit
    while len(gates) > 0:
        # Check 2-character gates first
        next_token = gates[:2]
        if next_token in gate_defs:
            circ.append_gate(gate_defs[next_token], (0,))
            gates = gates[2:]
        else:
            # Otherwise, check 1-character gates
            next_token = gates[0]
            if next_token in gate_defs:
                circ.append_gate(gate_defs[next_token], (0,))
            else:
                print("Unknown: ", next_token)
            gates = gates[1:]
    # Return the circuit
    return circ

def get_approx_t_str(angle: float, precision: int) -> str:
    # Divid angle by pi and mod by 2
    mod_angle = (angle / np.pi) % 2.0
    tol = 10.0 ** (-precision)

    # Edge errors, does not include Z for some reason
    if np.allclose(mod_angle, 0, atol=tol, rtol=0):
        return "I"
    elif np.allclose(mod_angle, 2, atol=tol, rtol=0):
        return "I"
    elif np.allclose(mod_angle, 1.25, atol=tol, rtol=0):
        return "ZT"
    elif np.allclose(mod_angle, 1.5, atol=tol, rtol=0):
        return "ZS"
    elif np.allclose(mod_angle, 0.75, atol=tol, rtol=0):
        return "ZTd"
    elif np.allclose(mod_angle, 1.75, atol=tol, rtol=0):
        return "Td"
    
    # print("Orig Angle: ", orig_angle, " Angle: ", angle, flush=True)
    num, den = Fraction(mod_angle).as_integer_ratio()
    # print("Angle: ", angle, "Num: ", num, "Den: ", den)
    return approximate_rz_direct(num, den, precision)[0]


def get_rz_perturbation_params(starting_angle, 
                               epsilon: int, 
                               num_trials: int = 5) -> tuple[list[RealVector],
                                                                list[float]]:
    
    default_params = ([[starting_angle, min(epsilon*2, MIN_EPSILON), 0]], [1])
    for _ in range(num_trials):
        params, probs = get_rz_perturbation_params_inner(starting_angle, epsilon)
        if params is not None:
            return params, probs
        
    return default_params

def get_rz_perturbation_params_inner(starting_angle, 
                                     epsilon: int) -> tuple[list[RealVector],
                                                                list[float]]:
    '''
    Return a list of params 
    '''
    if epsilon > MIN_EPSILON:
        # Try with epsilon
        epsilon = MIN_EPSILON

    final_params = [[starting_angle, epsilon, 0]]

    V = RZGate().get_unitary([starting_angle])
    U_1_t_str = get_approx_t_str(starting_angle, epsilon)
    U_1_t_circ = gridsynth_gates_to_cir(U_1_t_str)
    U_1 = U_1_t_circ.get_unitary()
    del U_1_t_circ

    Vt_U_1 = V.conj().T @ U_1

    del U_1

    # Expand to Pauli Basis and get Z component
    H = unitary_log_no_i(Vt_U_1)
    coeffs = pauli_expansion(H)
    az = coeffs[-1]
    del H
    del coeffs
    del Vt_U_1

    float_epsilon = 10 ** (-epsilon)
    delta = 2 * np.arcsin(np.sqrt(float_epsilon) / 2)
    if az < 0:
        delta = -delta
    
    final_params.append([starting_angle + delta, epsilon, 0])

    U_2_t_str = get_approx_t_str(starting_angle + delta, epsilon)
    U_2_t_circ = gridsynth_gates_to_cir(U_2_t_str)
    U_2 = U_2_t_circ.get_unitary()
    del U_2_t_circ

    # Twirl U_1
    final_params.append([starting_angle, epsilon, 1])
    U_3_t_str = "Z" + U_1_t_str + "Z"
    # Now do with U_2
    final_params.append([starting_angle + delta, epsilon, 1])
    U_4_t_str = "Z" + U_2_t_str + "Z"
    
    perturbed_t_strs = [U_1_t_str, U_2_t_str, U_3_t_str, U_4_t_str]
    perturbed_circs = [gridsynth_gates_to_cir(t_str) for t_str in perturbed_t_strs]

    [fix_phase(circ, V) for circ in perturbed_circs]
    perturbed_unitaries = [c.get_unitary() for c in perturbed_circs]
    del perturbed_circs
    perturbed_costs = [gp_frobenius_cost(u, V) for u in perturbed_unitaries]
    eps = np.mean(perturbed_costs)
    del perturbed_costs

    Vt_U_2 = V.conj().T @ U_2
    del U_2
    H = unitary_log_no_i(Vt_U_2)
    coeffs = pauli_expansion(H)
    Bz = coeffs[-1]

    del H
    del coeffs
    del Vt_U_2

    q = az / (az - Bz)

    p2 = q / 2
    p1 = (1 - q) / 2

    if np.allclose(p1, 0, atol=1e-5):
        p1 = 0
        p2 = 0.5
    elif np.allclose(p2, 0, atol=1e-5):
        p1 = 0.5
        p2 = 0

    probs = [p1, p2] * 2

    mean_un = np.average(perturbed_unitaries, axis=0, weights=probs)
    cost_of_mean = gp_frobenius_cost(mean_un, V)
    del perturbed_unitaries
    del mean_un
    ratio = cost_of_mean / eps / eps
    if ratio > 15 or p1 < 0 or p2 < 0:
        return None, None
    return final_params, probs

class GridSynthGate(QubitGate, CachedClass):
    """
    A gate representing an arbitrary rotation around the Z axis.

    It is given by the following parameterized unitary:

    .. math::

        \\begin{pmatrix}
        \\exp({-i\\frac{\\theta}{2}}) & 0 \\\\
        0 & \\exp({i\\frac{\\theta}{2}}) \\\\
        \\end{pmatrix}
    """

    _num_qudits = 1
    _num_params = 3
    _qasm_name = 'gg'
    # cache = LRUCache(maxsize=10)
    def get_unitary(self, params: RealVector = []) -> UnitaryMatrix:
        """Return the unitary for this gate, see :class:`Unitary` for more."""
        t_circ = self.get_circuit(params)
        un = t_circ.get_unitary()
        return un

    # def get_unitary(self, params: RealVector = []) -> UnitaryMatrix:
    #     """Return the unitary for this gate, see :class:`Unitary` for more."""
    #     # print(params, flush=True)
    #     angle = round(params[0], 20)
    #     epsilon = int(params[1])
    #     z_twirl = int(params[2])
    #     un = GridSynthGate.cache.get((angle, epsilon), None)
    #     if un is None:
    #         t_circ = self.get_circuit([angle, epsilon, 0])
    #         un = t_circ.get_unitary()
    #         GridSynthGate.cache[(angle, epsilon)] = un
    #     else:
    #         # print("Cache Hit!", flush=True)
    #         pass

    #     if z_twirl == 1:
    #         un = ZGate().get_unitary() @ un @ ZGate().get_unitary()

    #     return un

    def get_circuit(self, params: RealVector = []) -> Circuit:
        # self.check_parameters(params)
        angle = round(params[0], 20)
        epsilon = int(params[1])
        z_twirl = int(params[2])
        # If Z twirl is 0, no twirl
        t_str = get_approx_t_str(angle, epsilon)
        # If Z twirl is 1, add a Z gate
        if z_twirl == 1:
            t_str = "Z" + t_str + "Z"

        t_circ = gridsynth_gates_to_cir(t_str)
        return t_circ
    
    def get_qasm(self, params: RealVector, location: CircuitLocation) -> str:
        """Returns the qasm string for this gate."""
        return "{}({:.20e}, {}, {}) q[{}];\n".format(
            self.qasm_name,
            params[0],
            int(params[1]),
            int(params[2]),
            location[0],
        ).replace('()', '')
        
gg_gate_def = GateDef("gg", 3, 1, GridSynthGate())