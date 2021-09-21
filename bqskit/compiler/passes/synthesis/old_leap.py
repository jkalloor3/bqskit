"""This module implements the LEAPSynthesisPass."""
from __future__ import annotations

from bqskit.compiler.passes.synthesis.synthesis import SynthesisPass
from bqskit.qis.unitary import UnitaryMatrix
from typing import Any
from bqskit.ir import Circuit
from bqskit.ir.lang.qasm2 import OPENQASM2Language
from qsearch import options, leap_compiler, post_processing, assemblers, gatesets
#from qsearch import (
#    options, leap_compiler, post_processing, assemblers, gatesets,
#    multistart_solvers, parallelizers
#)

class OldLeap(SynthesisPass):

    def synthesize(self, utry: UnitaryMatrix, data: dict[str, Any]) -> Circuit:

        utry = utry.get_numpy()

        machine = data["machine_model"] if "machine_model" in data else None
        if machine is None:
            raise NotImplementedError(
                "Need to provide a machine model to synthesis"
            )
        adjacency_list = list(machine.coupling_graph)

        # Pass options into qsearch, being maximally quiet,
        # and set the target to utry
        opts = options.Options()
        opts.target = utry
        opts.gateset = gatesets.QubitCNOTAdjacencyList(adjacency_list)
        opts.verbosity = 1
        opts.write_to_stdout = True
        opts.reoptimize_size = 7

        # use the LEAP compiler, which scales better than normal qsearch
        compiler = leap_compiler.LeapCompiler()
        output = compiler.compile( opts )

        # LEAP requires some post-processing
        pp = post_processing.LEAPReoptimizing_PostProcessor()
        output = pp.post_process_circuit( output, opts )
        output = assemblers.ASSEMBLER_IBMOPENQASM.assemble( output )
        return OPENQASM2Language().decode(output)

#def synthesize_unitary(self, utry: UnitaryMatrix, data: dict[str, Any]) -> str:
#def synthesize_unitary(utry: UnitaryMatrix, data: dict[str, Any]) -> str:
#
#    utry = utry.get_numpy()
#
#    machine = data["machine_model"] if "machine_model" in data else None
#    if machine is None:
#        raise NotImplementedError(
#            "Need to provide a machine model to synthesis"
#        )
#    adjacency_list = list(machine.coupling_graph)
#
#    # Pass options into qsearch, being maximally quiet,
#    # and set the target to utry
#    opts = options.Options()
#    opts.verbosity = 0
#    opts.min_depth = 1
#    opts.target = utry
#    opts.reoptimize_size = 7
#    opts.timeout = 60*180
#    opts.threshold = 1e-14
#    opts.write_to_stdout = False
#    opts.weight_limit = 100
#    #opts.reoptimize_size = 7
#
#    opts.gateset = gatesets.QubitCNOTAdjacencyList(adjacency_list)
#
#    # LEAP requires some post-processing
#    opts.solver = multistart_solvers.MultiStart_Solver(8)
#    opts.parallelizer = parallelizers.ProcessPoolParallelizer
#
#    # use the LEAP compiler, which scales better than normal qsearch
#    compiler = leap_compiler.LeapCompiler()
#    output = compiler.compile( opts )
#
#    qasm_str = assemblers.ASSEMBLER_IBMOPENQASM.assemble(output)
#
#    return qasm_str
#