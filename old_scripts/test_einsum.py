import numpy as np
from util import load_block, load_jiggled_ensemble, create_jiggled_unitaries
from bqskit.compiler import Compiler
from bqskit.ir.circuit import Circuit, CircuitGate
from bqskit.passes import ScanPartitioner

def create_arrays(ensemble: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    M = ensemble.shape[0]
    tr_V_Us = np.zeros((M, ), dtype=np.complex128)
    tr_Us = np.zeros((M, M), dtype=np.complex128)

    for i, un in enumerate(ensemble):
        trace_dist = np.trace(un @ target.conj().T)
        tr_V_Us[i] = trace_dist

    for i, un in enumerate(ensemble):
        for j, un2 in enumerate(ensemble):
            trace_dist = np.trace(un.conj().T @ un2)
            tr_Us[i, j] = trace_dist


    return tr_V_Us, tr_Us

qasms_file = "/pscratch/sd/j/jkalloor/bqskit/small_block_checkpoints_final_paper_4_clifft_tket/draper_adder_12_0_3.0/block_0/ensemble_0_.qasms"
jiggle_file = "small_block_checkpoints_final_paper_4_clifft_tket/draper_adder_12_0_3.0/block_0/ensemble_0_jiggles_.npy"
cache_file = "/pscratch/sd/j/jkalloor/bqskit/small_block_checkpoints_final_paper_4_clifft_tket/draper_adder_12_0_3.0/block_0/ensemble_0_cache.pkl"



def create_arrays_2(ensemble: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Create arrays for einsum operations."""
    M = ensemble.shape[0]
    tr_V_Us = np.einsum("mij,ij->m", ensemble, target.conj(), optimize=True)
    tr_Us = np.einsum("aij,bij->ab", ensemble.conj(), ensemble, optimize=True)
    return tr_V_Us, tr_Us


if __name__ == '__main__':

    orig_circ = load_block("draper_adder_12", "0", "_tket")
    orig_circ = Circuit.from_file(orig_circ)
    print(orig_circ.gate_counts, flush=True)

    compiler = Compiler(num_workers=1)
    part_circ = compiler.compile(orig_circ, [ScanPartitioner(4)])

    block_target_circs = {}

    for i, op in enumerate(part_circ.operations()):
        assert isinstance(op.gate, CircuitGate)
        block_target_circs[str(i)] = op.gate._circuit

    target_circ: Circuit = block_target_circs['0']
    target = target_circ.get_unitary()

    circ_params = load_jiggled_ensemble(qasms_file, jiggle_file, cache_file)

    ensemble = []
    for circ_param in circ_params:
        ensemble.extend(create_jiggled_unitaries(circ_param, target=target))


    dists = [d for _, d in ensemble]
    uns = [u for u, _ in ensemble]
    ensemble_uns = np.array(uns)

    tr_v_us, tr_us = create_arrays_2(ensemble_uns, target)
    tr_v_us_1, tr_us_1 = create_arrays(ensemble_uns, target)

    assert np.allclose(tr_v_us, tr_v_us_1)
    assert np.allclose(tr_us, tr_us_1)