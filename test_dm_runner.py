import numpy as np
import time
import pickle

from util.common import load_circuit

from util.dm_runner import generate_full_runner, DensityMatrixRunner, create_large_block_runner
from util.samplers import get_file_names
from util.unitary_dm_pass import calculate_good_blocks, update_partitioned_data
from util.distance import trace_distance
from bqskit.qis import StateVector
from bqskit.ir.circuit import Circuit
from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData

from bqskit.compiler import Compiler


partitioned_data_file = f"partitioned_data_all_circs_clifft.pickle"
all_partitioned_data = pickle.load(open(partitioned_data_file, "rb"))


# Testing with bqskit compiler

class TestRunner(BasePass):

    async def run(self, circuit: Circuit, data: PassData) -> None:
        """Test the DensityMatrixRunner within a pass."""

        # await test_large_runner("mult8")
        await test_circ_runner("FermiHubbard2x2_jw_long", 2.0)


async def test_circ_runner(circ_name: str = "LiH_jw_long", 
                      max_tol: float = 2.0):
    checkpoint_form = "small_block_checkpoints_final_paper_4_clifft_tket/{circ_name}_{large_block_num}_{tol}"
    good_blocks = calculate_good_blocks(
        circ_name=circ_name,
        checkpoint_form=checkpoint_form,
        max_tol=max_tol,
        cliff_t=True
    )
    # Just use 2 of the small blocks for each large block
    for large_block_num in good_blocks:
        good_blocks[large_block_num] = set(list(good_blocks[large_block_num])[:2])

    updated_data = update_partitioned_data(
        partitioned_data=all_partitioned_data[circ_name],
        good_blocks=good_blocks
    )

    print("Updated Data Keys: ", list(updated_data.keys()))

    orig_circ: Circuit = load_circuit(circ_name=circ_name)
    num_qubits = orig_circ.num_qudits
    runner = generate_full_runner(
        circ_name=circ_name,
        max_tol=max_tol,
        partitioned_data=updated_data,
        checkpoint_form=checkpoint_form,
        partitioned_circ_file=f"partitioned_circs/{circ_name}.pickle",
        cliff_t=True
    )

    init_sv = StateVector.random(orig_circ.num_qudits)
    rho_in = np.outer(init_sv.numpy, np.conj(init_sv.numpy))
    # Get original unitary for comparison
    original_unitary = orig_circ.get_unitary()
    rho_orig = original_unitary @ rho_in @ np.conj(original_unitary.T)
    rho_out = await runner.run(rho_in)
    # Calculate trace distance of rhos
    ens_eps = trace_distance(rho_out, rho_orig)
    print("Ensemble Trace Distance: ", ens_eps)


async def test_large_runner(circ_name: str = "LiH_jw_long", 
                      large_block_num: str = "0", 
                      max_tol: float = 2.0):

    checkpoint_form = "small_block_checkpoints_final_paper_4_clifft_tket/{circ_name}_{large_block_num}_{max_tol}"
    good_blocks = calculate_good_blocks(
        circ_name=circ_name,
        checkpoint_form=checkpoint_form,
        max_tol=max_tol,
        cliff_t=True
    )

    # Just use 2 of the small blocks
    good_blocks[large_block_num] = set(list(good_blocks[large_block_num])[:2])
    

    updated_data = update_partitioned_data(
        partitioned_data=all_partitioned_data[circ_name],
        good_blocks=good_blocks
    )

    print("Updated Data Keys: ", list(updated_data[large_block_num][0].keys()))

    orig_circ: Circuit = all_partitioned_data[circ_name][large_block_num][1]
    num_qubits = orig_circ.num_qudits

    runner = create_large_block_runner(
        qubits=np.arange(num_qubits),
        circ_name=circ_name,
        large_block_num=large_block_num,
        max_tol=max_tol,
        partitioned_data=updated_data[large_block_num],
        checkpoint_form=checkpoint_form,
        cliff_t=True
    )

    init_sv = StateVector.random(orig_circ.num_qudits)
    rho_in = np.outer(init_sv.numpy, np.conj(init_sv.numpy))

    # Get original unitary for comparison
    original_circ = updated_data[large_block_num][1]
    original_unitary = original_circ.get_unitary()
    rho_orig = original_unitary @ rho_in @ np.conj(original_unitary.T)


    # Get a random unitary from runner sampler to compare against
    # u, _ = next(runner.sampler)
    # sample_rho_out = u @ rho_in @ np.conj(u.T)
    # runner.sampler.reset()

    # Calculate trace distance of rhos
    # eps = trace_distance(sample_rho_out, rho_orig)

    rho_out = await runner.run(rho_in)

    # Calculate trace distance of rhos
    # eps = trace_distance(sample_rho_out, rho_orig)
    ens_eps = trace_distance(rho_out, rho_orig)

    print("Ensemble Trace Distance: ", ens_eps)



def test_small_runner(circ_name: str = "heisenberg7", 
                      large_block_num: str = "0", 
                      max_tol: float = 2.0):

    sub_block_num = "00"
    ensemble_file_names = get_file_names(
        large_checkpoint_dir=f"small_block_checkpoints_final_paper_4_clifft_tket/{circ_name}_{large_block_num}_{max_tol}",
        small_block_num=sub_block_num
    )[:-1]

    start = time.time()


    runner = DensityMatrixRunner(
        qubits=[0,1,2,3],
        ensemble_file_names=ensemble_file_names,
        cliff_t=True
    )

    init_sv = StateVector.random(4)
    rho_in = np.outer(init_sv.numpy, np.conj(init_sv.numpy))

    # Get original unitary for comparison
    original_circ = all_partitioned_data[circ_name][large_block_num][0][sub_block_num]
    original_unitary = original_circ.get_unitary()
    rho_orig = original_unitary @ rho_in @ np.conj(original_unitary.T)


    # Get a random unitary from runner sampler to compare against
    u, _ = next(runner.sampler)
    sample_rho_out = u @ rho_in @ np.conj(u.T)
    runner.sampler.reset()

    rho_out = runner.run(rho_in)


    final_time = time.time() - start

    # Calculate trace distance of rhos
    eps = trace_distance(sample_rho_out, rho_orig)
    ens_eps = trace_distance(rho_out, rho_orig)

    print("Final Time: ", final_time)
    print("Trace Distance: ", eps)
    print("Ensemble Trace Distance: ", ens_eps)

if __name__ == '__main__':
    # test_small_runner()
    # test_large_runner(circ_name="mult8")

    compiler = Compiler(num_workers=128)
    circ = Circuit(2)

    start = time.time()
    compiler.compile(circuit=circ, workflow=[TestRunner()])
    print("Total Time: ", time.time() - start)