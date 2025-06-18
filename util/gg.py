"""This module implements the RZGate."""
from __future__ import annotations

import numpy as np

# from multiprocessing import shared_memory, Lock
from typing import TYPE_CHECKING

from bqskit.ir.circuit import Circuit, CircuitLocation
from bqskit.ir.gates import (IdentityGate, ZGate, SGate, SdgGate, 
                            TGate, HGate, TdgGate, XGate, RZGate, YGate)
    
from bqskit.ir.gates.qubitgate import QubitGate
from bqskit.qis.unitary.unitary import RealVector
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.utils.cachedclass import CachedClass
from bqskit.qis.pauli import PauliMatrices
from pyLIQTR.gate_decomp.gate_approximation import approximate_rz_direct

from .fix_global_phase import fix_phase
from .distance import gp_frobenius_cost
from fractions import Fraction
from bqskit.runtime import get_runtime

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
    # tol = 10 ** (-precision)
    tol = pow(10.0, -precision)

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

def get_az(H):
    """
    Computes a Pauli expansion of the hermitian matrix H and returns the Z 
    coefficient.
    """

    # Change basis of H to Pauli Basis (solve for coefficients -> X)
    n = int(np.log2(len(H)))
    paulis = PauliMatrices(n)
    flatten_paulis = [np.reshape(pauli, 4 ** n) for pauli in paulis]
    flatten_H = np.reshape(H, 4 ** n)
    A = np.stack(flatten_paulis, axis=-1)
    X = np.matmul(np.linalg.inv(A), flatten_H)

    # Want to make the first coefficient real
    global_phase = np.angle(X[0])
    X = np.exp(-1j * global_phase) * X

    # Return imaginary part of the last coefficient
    return np.imag(X[-1])


def get_rz_perturbations(starting_angle, 
                         epsilon: int,
                         target_angle: float = None) -> tuple[list[RealVector], 
                                                list[str], list[float]]:
    '''
    Returns a list of 4 parameters and a list of strings to put in the the
    cache of the requester. Furthermore, returns the probabilities of each
    parameter.
    '''
    if epsilon > MIN_EPSILON:
        # Try with epsilon
        epsilon = MIN_EPSILON

    if target_angle is None:
        target_angle = starting_angle

    V = RZGate().get_unitary([target_angle])
    U_1_t_str = get_approx_t_str(starting_angle, epsilon)
    U_1_t_circ = gridsynth_gates_to_cir(U_1_t_str)
    U_1 = U_1_t_circ.get_unitary()

    final_strs = []
    final_params = []
    final_params.append([starting_angle, epsilon, 0])

    final_strs.append(U_1_t_str)

    Vt_U_1 = V.conj().T @ U_1

    az = get_az(Vt_U_1)

    float_epsilon = 10.0 ** (-epsilon)
    delta = np.arcsin(float_epsilon)
    if az < 0:
        delta = -delta

    Bz = az
    while az == Bz:
        # Keep modifying delta until we get different gridsynth approximations
        delta = delta * 2
        U_2_t_str = get_approx_t_str(starting_angle + delta, epsilon)
        U_2_t_circ = gridsynth_gates_to_cir(U_2_t_str)
        U_2 = U_2_t_circ.get_unitary()

        Vt_U_2 = V.conj().T @ U_2
        Bz = get_az(Vt_U_2)

    final_params.append([starting_angle + delta, epsilon, 0])
    final_strs.append(U_2_t_str)

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
    final_params.append([starting_angle, epsilon, 1])
    final_params.append([starting_angle + delta, epsilon, 1])
    return final_params, final_strs, probs

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
    def get_unitary(self, params: RealVector = []) -> UnitaryMatrix:
        """Return the unitary for this gate, see :class:`Unitary` for more."""
        ang = round(params[0], 18)
        epsilon = int(params[1])
        z_twirl = int(params[2])
        ind = (ang, epsilon)
        try:
            cache = get_runtime().get_cache()
            t_str = cache.get(ind, None)
        except:
            t_str = None

        if t_str is None:
            t_str = get_approx_t_str(ang, epsilon)            
        if z_twirl == 1:
            t_str = "Z" + t_str + "Z"
        
        un = gridsynth_gates_to_cir(t_str).get_unitary()
        return un
    
    def get_unitary_w_cache(self, params: RealVector = [], cache: dict = {}) -> UnitaryMatrix:
        """Return the unitary for this gate, see :class:`Unitary` for more."""
        ang = round(params[0], 18)
        epsilon = int(params[1])
        z_twirl = int(params[2])
        ind = (ang, epsilon)
        t_str = cache.get(ind, None)

        if t_str is None:
            print("Cache miss for: ", ind, flush=True)
            t_str = get_approx_t_str(ang, epsilon)            
        if z_twirl == 1:
            t_str = "Z" + t_str + "Z"
        
        un = gridsynth_gates_to_cir(t_str).get_unitary()
        return un

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