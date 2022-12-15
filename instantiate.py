# %%
"""
Numerical Instantiation is the foundation of many of BQSKit's algorithms.

This example demonstrates building a circuit template that can implement the
toffoli gate and then instantiating it to be the gate.
"""
from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import unitary_group

from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import VariableUnitaryGate
from bqskit.ir.gates.constant.cx import CNOTGate
from bqskit.ir.gates.parameterized.u3 import U3Gate
from bqskit.ir.gates.parameterized.unitary_acc import VariableUnitaryGateAcc
from bqskit.qis.unitary import UnitaryMatrix
jax.config.update('jax_enable_x64', True)


from bqskit import enable_logging

# enable_logging(True)
use_jax = True
use_jax = False
multistarts = 32

for use_jax in [True, False]:
    print(f"*******************{use_jax}***********************")


    if use_jax:
        matlib = jnp
    else:
        matlib = np

    # We will optimize towards the toffoli unitary.
    toffoli = matlib.array([
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 1, 0],
    ])


    # Start with the circuit structure


    if use_jax:
        gate = VariableUnitaryGateAcc
    else:
        gate = VariableUnitaryGate

    gate = VariableUnitaryGate

    for _ in range(3):
        circuit = Circuit(3)


        for _ in range(1):
            # circuit.append_gate(U3Gate(), [1])
            # circuit.append_gate(CNOTGate(), [0, 1])
            circuit.append_gate(gate(1), [0])
            circuit.append_gate(gate(1), [1])
            circuit.append_gate(gate(1), [2])
            circuit.append_gate(gate(2), [1, 2])
            # circuit.append_gate(gate(2), [0, 2])
            # circuit.append_gate(gate(2), [1, 2])
            # circuit.append_gate(gate(1), [0])
            # circuit.append_gate(gate(1), [1])
            # circuit.append_gate(gate(1), [2])
            circuit.append_gate(gate(2), [0, 2])
            circuit.append_gate(gate(2), [0, 1])



        # target = UnitaryMatrix(unitary_group.rvs(2**7), use_jax=True, check_arguments=False)
        target = unitary_group.rvs(2**7)
        target = toffoli
        gates_location = tuple([
            (1, 2, 3),
            (0, 2),
            (1, 5),
            (0, 2),
            (0, 4,5),
            (4, 6),
            (0, 2),
            (5, 6),
            (0, 2),
            (1, 5),
            (0, 2),
            (0, 4),
            (4, 6),
            (0, 2),
            (1, 4),
        ])

        big_circuit = Circuit(7)
        for loc in gates_location:
            big_circuit.append_gate(gate(len(loc)), loc)

        # circuit = big_circuit

        if use_jax:
            method = 'qfactor_jax_batched_jit'
        else:
            method='qfactor'
        
        # %%
        # Instantiate the circuit template with qfactor
        tic = time.perf_counter()
        # jax.profiler.start_trace("/global/homes/a/alonkukl/Repos/bqskit/trace_logs/trace_toffoli")
        # jax.profiler.start_server(9999)
        # time.sleep(15)

        circuit.instantiate(
            target,
            method=method,
            multistarts=multistarts,
            diff_tol_a=1e-11,   # Stopping criteria for distance change
            diff_tol_r=1e-5,    # Relative criteria for distance change
            dist_tol=1e-11,     # Stopping criteria for distance
            max_iters=10000,   # Maximum number of iterations
            min_iters=100,     # Minimum number of iterations
            # slowdown_factor=0,   # Larger numbers slowdown optimization
            # to avoid local minima
        )
        # time.sleep(100)

        # jax.profiler.stop_trace()

        toc = time.perf_counter()

        print(f'Using {method} it took {toc-tic} seconeds for {multistarts} multistarts')

        # Calculate and print final distance
        dist = circuit.get_unitary().get_distance_from(target, 1)
        print('Final Distance: ', dist)

    # You can use synthesis to convert the `VariableUnitaryGate`s to
    # native gates. Alternatively, you can build a circuit directly out of
    # native gates and use the default instantiater to instantiate directly.

# %%
