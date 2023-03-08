from __future__ import annotations
import functools

import logging
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from scipy.stats import unitary_group

import os

from bqskit.ir.gates.constantgate import ConstantGate
from bqskit.ir.gates.parameterized.u3 import U3Gate
from bqskit.ir.gates.parameterized.unitary import VariableUnitaryGate
from bqskit.ir.gates.parameterized.unitary_acc import VariableUnitaryGateAcc
from bqskit.ir.opt.instantiater import Instantiater
from bqskit.qis.state.state import StateVector
from bqskit.qis.unitary.unitarybuilderjax import UnitaryBuilderJax
from bqskit.qis.unitary.unitarymatrix import UnitaryMatrix
from bqskit.qis.unitary.unitarymatrixjax import UnitaryMatrixJax
import time


if TYPE_CHECKING:
    from bqskit.ir.circuit import Circuit


_logger = logging.getLogger(__name__)

jax.config.update('jax_enable_x64', True)

class QFactor_jax_batched_jit(Instantiater):
    """The QFactor batch circuit instantiater."""

    def __init__(
        self,
        diff_tol_a: float = 1e-12,
        diff_tol_r: float = 1e-6,
        dist_tol: float = 1e-10,
        max_iters: int = 100000,
        min_iters: int = 1000,
    ):

        if not isinstance(diff_tol_a, float) or diff_tol_a > 0.5:
            raise TypeError('Invalid absolute difference threshold.')

        if not isinstance(diff_tol_r, float) or diff_tol_r > 0.5:
            raise TypeError('Invalid relative difference threshold.')

        if not isinstance(dist_tol, float) or dist_tol > 0.5:
            raise TypeError('Invalid distance threshold.')

        if not isinstance(max_iters, int) or max_iters < 0:
            raise TypeError('Invalid maximum number of iterations.')

        if not isinstance(min_iters, int) or min_iters < 0:
            raise TypeError('Invalid minimum number of iterations.')

        self.diff_tol_a = diff_tol_a
        self.diff_tol_r = diff_tol_r
        self.dist_tol = dist_tol
        self.max_iters = max_iters
        self.min_iters = min_iters

    def instantiate(
        self,
        circuit,  # : Circuit,
        target: UnitaryMatrix | StateVector,
        x0,
    ):

        return self.instantiate_multistart(circuit, target, [x0])

    def instantiate_multistart(
        self,
        circuit,  # : Circuit,
        target: UnitaryMatrix | StateVector,
        starts: list[npt.NDArray[np.float64]],
    ):
        if len(circuit) == 0:
            return np.array([])

        in_c = circuit
        circuit = circuit.copy()
        
        # A very ugly casting
        for op in circuit:
            g = op.gate
            if isinstance(g, VariableUnitaryGate):
                g.__class__ = VariableUnitaryGateAcc

        """Instantiate `circuit`, see Instantiater for more info."""
        target = UnitaryMatrixJax(target)
        amount_of_starts = len(starts)
        locations = tuple([op.location for op in circuit])
        gates = tuple([op.gate for op in circuit])
        biggest_gate_size = max(gate.num_qudits for gate in gates)

        untrys = []

        for gate in gates:
            size_of_untry = 2**gate.num_qudits
            
            if isinstance(gate, VariableUnitaryGateAcc):
                untry =  unitary_group.rvs(size_of_untry)
            else:
                untry = gate.get_unitary().numpy

            untrys.append([
                _apply_padding_and_flatten(
                    untry , gate, biggest_gate_size,
                ) for _ in range(amount_of_starts)
            ])

        untrys = jnp.array(np.stack(untrys, axis=1))
        n = 40
        plateau_windows_size = 8
        res_var = _sweep2_jited(
                target, locations, gates, untrys, n, self.dist_tol, self.diff_tol_a, self.diff_tol_r, plateau_windows_size, self.max_iters, self.min_iters, amount_of_starts
            )
        best_start = 0
        it = res_var["iteration_counts"][0]
        c1s = res_var["c1s"]
        untrys = res_var["untrys"]

        # for p in res_var["plateau_windows"]:
            # print(p[:it])

        if any(res_var["curr_reached_required_tol_l"]):
            best_start = jnp.argmin(jnp.abs(c1s))
            _logger.info(
                f'Terminated: {it} c1 = {c1s} <= dist_tol.\n Best start is {best_start}',
            )
        elif all(res_var["curr_plateau_calc_l"]):
            _logger.info(
                    f'Terminated: |c1 - c2| = '
                    ' <= diff_tol_a + diff_tol_r * |c1|.',
                )
            best_start = jnp.argmin(jnp.abs(c1s))

            _logger.info(
                f'Terminated: {it} c1 = {c1s} Reached plateuo.\n Best start is {best_start}',
                )
        elif it >= self.max_iters:
            _logger.info('Terminated: iteration limit reached.')
            best_start = jnp.argmin(jnp.abs(c1s))
        else:
            _logger.error(f'Terminated with no good reason after {it} iterstion with c1s {c1s}.')
        params = []
        for untry, gate in zip(untrys[best_start], gates):
            params.extend(
                gate.get_params(
                _remove_padding_and_create_matrix(untry, gate),
                ),
            )
        return np.array(params)

    @staticmethod
    def get_method_name() -> str:
        """Return the name of this method."""
        return 'qfactor_jax_batched_jit'

    @staticmethod
    def can_internaly_perform_multistart() -> bool:
        """Probes if the instantiater can internaly perform multistrat."""
        return True

    @staticmethod
    def is_capable(circuit) -> bool:
        """Return true if the circuit can be instantiated."""
        return all(
            isinstance(gate, (VariableUnitaryGate, VariableUnitaryGateAcc, U3Gate, ConstantGate))
            for gate in circuit.gate_set
        )

    @staticmethod
    def get_violation_report(circuit) -> str:
        """
        Return a message explaining why `circuit` cannot be instantiated.

        Args:
            circuit (Circuit): Generate a report for this circuit.

        Raises:
            ValueError: If `circuit` can be instantiated with this
                instantiater.
        """

        invalid_gates = {
            gate
            for gate in circuit.gate_set
            if not isinstance(gate, (VariableUnitaryGate, VariableUnitaryGateAcc, U3Gate, ConstantGate))
        }

        if len(invalid_gates) == 0:
            raise ValueError('Circuit can be instantiated.')

        return (
            'Cannot instantiate circuit with qfactor'
            ' because the following gates are not locally optimizable with jax: %s.'
            % ', '.join(str(g) for g in invalid_gates)
        )


def _initilize_circuit_tensor(
    target_num_qudits,
    target_radixes,
    locations,
    target_mat,
    untrys,
):

    target_untry_builder = UnitaryBuilderJax(
        target_num_qudits, target_radixes, target_mat.conj().T,
    )

    for loc, untry in zip(locations, untrys):
        target_untry_builder.apply_right(
            untry, loc, check_arguments=False,
        )

    return target_untry_builder


def _single_sweep(locations, gates, amount_of_gates, target_untry_builder, untrys):
    # from right to left
    for k in reversed(range(amount_of_gates)):
        gate = gates[k]
        location = locations[k]
        untry = untrys[k]

        # Remove current gate from right of circuit tensor
        target_untry_builder.apply_right(
            untry, location, inverse=True, check_arguments=False,
        )

        # Update current gate
        if gate.num_params > 0:
            env = target_untry_builder.calc_env_matrix(location)
            untry = gate.optimize(env, get_untry=True)
            untrys[k] = untry

            # Add updated gate to left of circuit tensor
        target_untry_builder.apply_left(
            untry, location, check_arguments=False,
        )

        # from left to right
    for k in range(amount_of_gates):
        gate = gates[k]
        location = locations[k]
        untry = untrys[k]

        # Remove current gate from left of circuit tensor
        target_untry_builder.apply_left(
            untry, location, inverse=True, check_arguments=False,
        )

        # Update current gate
        if gate.num_params > 0:
            env = target_untry_builder.calc_env_matrix(location)
            untry = gate.optimize(env, get_untry=True)
            untrys[k] = untry

            # Add updated gate to right of circuit tensor
        target_untry_builder.apply_right(
            untry, location, check_arguments=False,
        )

    return target_untry_builder, untrys


def _apply_padding_and_flatten(untry, gate, max_gate_size):
    zero_pad_size = (2**max_gate_size)**2 - (2**gate.num_qudits)**2
    if zero_pad_size > 0:
        zero_pad = jnp.zeros(zero_pad_size)
        return jnp.concatenate((untry, zero_pad), axis=None)
    else:
        return jnp.array(untry.flatten())


def _remove_padding_and_create_matrix(untry, gate):
    len_of_matrix = 2**gate.num_qudits
    size_to_keep = len_of_matrix**2
    return untry[:size_to_keep].reshape((len_of_matrix, len_of_matrix))


def Loop_vars(untrys, c1s, plateau_windows, curr_plateau_calc_l, curr_reached_required_tol_l, iteration_counts, target_untry_builders):
    d = {}
    d["untrys"] = untrys
    d["c1s"] = c1s
    d["plateau_windows"] = plateau_windows
    d["curr_plateau_calc_l"] = curr_plateau_calc_l
    d["curr_reached_required_tol_l"] = curr_reached_required_tol_l
    d["iteration_counts"] = iteration_counts
    d["target_untry_builders"] = target_untry_builders

    return d

def _sweep2(target, locations, gates, untrys, n, dist_tol, diff_tol_a, diff_tol_r, plateau_windows_size, max_iters, min_iters, amount_of_starts):

    c1s = jnp.array([1.0] * amount_of_starts)
    plateau_windows = jnp.array([[0] * plateau_windows_size for _ in range(amount_of_starts)], dtype=bool)

    def should_continue(var):
        return jnp.logical_not(
                    jnp.logical_or(
                        jnp.any(var["curr_reached_required_tol_l"]), 
                        jnp.logical_or(
                            var["iteration_counts"][0] > max_iters,
                            jnp.logical_and(
                                var["iteration_counts"][0] > min_iters,
                                jnp.all(var["curr_plateau_calc_l"])))))
                                
    def _while_body_to_be_vmaped(untrys, c1, plateau_window, curr_plateau_calc, curr_reached_required_tol, iteration_count, target_untry_builder_tensor):

        amount_of_gates = len(gates)
        amount_of_qudits = target.num_qudits
        target_radixes = target.radixes

        untrys_as_matrixs = []
        for gate_index, gate in enumerate(gates):
            untrys_as_matrixs.append(
                UnitaryMatrixJax(
                    _remove_padding_and_create_matrix(
                        untrys[gate_index], gate,
                    ), gate.radixes,
                ),
            )
        untrys = untrys_as_matrixs

        ##### initilize every "n" iterations of the loop
        operand_for_if = (untrys, target_untry_builder_tensor)
        initilize_body = lambda x: _initilize_circuit_tensor(
            amount_of_qudits, target_radixes, locations, target.numpy, x[0]
        ).tensor
        no_initilize_body = lambda x: x[1]
        
        target_untry_builder_tensor = jax.lax.cond(iteration_count % n == 0, initilize_body, no_initilize_body, operand_for_if)


        target_untry_builder = UnitaryBuilderJax(amount_of_qudits, target_radixes, tensor=target_untry_builder_tensor)

        iteration_count  = iteration_count + 1

        target_untry_builder, untrys = _single_sweep(
            locations, gates, amount_of_gates, target_untry_builder, untrys,
        )


        c2 = c1
        dim = target_untry_builder.dim
        untry_res = target_untry_builder.tensor.reshape((dim, dim))
        c1 = jnp.abs(jnp.trace(untry_res))
        c1 = 1 - (c1 / (2 ** amount_of_qudits))

        curr_plateau_part = jnp.abs(c1 - c2) <= diff_tol_a + diff_tol_r * jnp.abs(c1)
        curr_plateau_calc = functools.reduce(jnp.bitwise_or, plateau_window) | curr_plateau_part
        plateau_window = jnp.concatenate((jnp.array([curr_plateau_part]), plateau_window[:-1]))
        curr_reached_required_tol = c1 < dist_tol


        biggest_gate_size = max(gate.num_qudits for gate in gates)
        final_untrys_padded = jnp.array([
            _apply_padding_and_flatten(
                untry.numpy.flatten(
                ), gate, biggest_gate_size,
            ) for untry, gate in zip(untrys, gates)
        ])

        return final_untrys_padded, c1, plateau_window, curr_plateau_calc, curr_reached_required_tol, iteration_count, target_untry_builder.tensor


    while_body_vmaped = jax.vmap(_while_body_to_be_vmaped)

    def while_body(var):
        return Loop_vars(*while_body_vmaped(var["untrys"], var["c1s"], var["plateau_windows"], var["curr_plateau_calc_l"], var["curr_reached_required_tol_l"], var["iteration_counts"], var["target_untry_builders"]))

    dim = np.prod(target.radixes)
    initial_untray_builders_values = jnp.array([jnp.identity(dim, dtype=jnp.complex128).reshape(target.radixes * 2) for _ in range(amount_of_starts)])

    initial_loop_var = Loop_vars(untrys, c1s, plateau_windows, jnp.array([False] * amount_of_starts),  jnp.array([False]*amount_of_starts), jnp.array([0]*amount_of_starts), initial_untray_builders_values)
    res_var = jax.lax.while_loop(should_continue, while_body, initial_loop_var)

    return res_var

if "NO_JIT_QFACTOR" in os.environ:
    _sweep2_jited = _sweep2
else:
    _sweep2_jited = jax.jit(_sweep2, static_argnums=(1, 2, 4, 5, 6, 7, 8, 9, 10, 11))
    


