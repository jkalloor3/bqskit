from bqskit.ir.circuit import Circuit
from bqskit.qis import StateVector, UnitaryMatrix, UnitaryBuilder
import numpy as np
import time
import pickle

from functools import reduce
from util.common import load_circuit

from util.dm_runner import generate_full_runner, DensityMatrixRunner, create_large_block_runner
from util.samplers import get_file_names, get_single_rho, apply_superoperator
from util.unitary_dm_pass import calculate_good_blocks, update_partitioned_data
from util.distance import trace_distance, fidelity
# from bqskit.ir.gates import ConstantUnitaryGate
from bqskit.compiler.basepass import BasePass
from bqskit.compiler.passdata import PassData

from bqskit.compiler import Compiler
from bqskit.runtime import get_runtime

CLIFF_T = True
cliff_t_string = "_clifft" if CLIFF_T else ""

partitioned_data_file = f"partitioned_data_all_circs{cliff_t_string}.pickle"
all_partitioned_data = pickle.load(open(partitioned_data_file, "rb"))

if CLIFF_T:
    print("Using cliff-t circuits", flush=True)
    checkpoint_form = "small_block_checkpoints_final_paper_4_clifft_tket/{circ_name}_{large_block_num}_{max_tol}"
else:
    print("Using non-cliff-t circuits", flush=True) 
    checkpoint_form = "small_block_checkpoints_final_paper_4_more_cx_tket/{circ_name}_{large_block_num}_{max_tol}"

def embed_unitary(u, qubits, n):
    builder = UnitaryBuilder(n)
    builder.apply_right(u, qubits.tolist())
    return builder.get_unitary()

def apply_unitary_normal(U, rho):
    """
    Returns vec(U rho U^dag) as a vector,
    or reshape back to a matrix if desired.
    """

    return np.einsum('ai,ij,bj->ab', U, rho, U.conj(), 
                     optimize=True, dtype=U.dtype)


# Testing with bqskit compiler
class TestRunner(BasePass):

    async def run(self, circuit: Circuit, data: PassData) -> None:
        """Test the DensityMatrixRunner within a pass."""

        # await test_small_runner()
        await test_large_runner("draper_adder_12", "0")
        # await test_circ_runner("FermiHubbard2x2_jw_long", 2.0)
        # await get_runtime().map(TestRunner.rho_comp_test, range(5))
        # for i in range(5):
        #     await TestRunner.rho_comp_test(i)
        # num_tests = 40
        # for num_qubits in range(12, 13):
        #     print(f"Testing {num_qubits} qubits")
        #     start = time.time()
        #     self.single_rho_test(num_qubits=num_qubits, num_test=num_tests)
        #     single_time = time.time() - start
        #     # # self.normal_rho_test(num_qubits=num_qubits, num_test=num_tests)
        #     # normal_time = time.time() - start - single_time
        #     start = time.time()
        #     self.superop_rho_test(num_qubits=num_qubits, num_test=num_tests)
        #     superop_time = time.time() - start
        #     print(f"Single Rho Time: {single_time}, Superop Rho Time: {superop_time}")

    def superop_rho_test(self, num_qubits: int = 6, num_test: int = 500):
        init_sv = StateVector.random(num_qubits)
        rho_in = np.outer(init_sv.numpy, np.conj(init_sv.numpy))
        gate_size = 4
        super_op = np.zeros((2**(2*gate_size), 
                             2**(2*gate_size)), dtype=np.complex128)

        start = time.time()

        for _ in range(num_test):
            rand_un = UnitaryMatrix.random(gate_size)
            # Choose `gate_size` qubits from `num_qubits`
            super_op += np.kron(rand_un.conj(), rand_un)

        end = time.time()
        print(f"Superoperator Construction Time: {end - start}")
        qubits = np.random.choice(num_qubits, size=gate_size, replace=False)
        final_rho = apply_superoperator(rho_in, super_op, num_qubits, qubits)
        return final_rho


    def single_rho_test(self, num_qubits: int = 6, num_test: int = 500):
        init_sv = StateVector.random(num_qubits)
        rho_in = np.outer(init_sv.numpy, np.conj(init_sv.numpy))
        final_rho = rho_in.copy()
        gate_size = 4
        qubits = np.random.choice(num_qubits, size=gate_size, replace=False)

        for _ in range(num_test):
            rand_un = UnitaryMatrix.random(gate_size)
            final_rho += get_single_rho(rand_un, rho_in, qubits)

        return final_rho

    def normal_rho_test(self, num_qubits: int = 6, num_test: int = 500):
        init_sv = StateVector.random(num_qubits)
        rho_in = np.outer(init_sv.numpy, np.conj(init_sv.numpy))
        final_rho = rho_in.copy()
        gate_size = 4
        qubits = np.random.choice(num_qubits, size=gate_size, replace=False)
        for _ in range(num_test):
            rand_un = UnitaryMatrix.random(gate_size)
            # Apply unitary on qubits by tensoring with identities
            full_un = embed_unitary(rand_un, qubits, num_qubits)
            rho_orig = full_un @ rho_in @ np.conj(full_un.T)
            final_rho += rho_orig

        return final_rho

    @staticmethod
    async def rho_comp_test(i: int) -> int:
        # Create a random unitary on up to 6 qubits
        num_qubits = 8
        gate_size = np.random.randint(3, 7)
        rand_un = UnitaryMatrix.random(gate_size)

        # Choose `gate_size` qubits from `num_qubits`
        qubits = np.random.choice(num_qubits, size=gate_size, replace=False)

        # Create a random initial state vector on up to 6 qubits
        init_sv = StateVector.random(num_qubits)

        rho_in = np.outer(init_sv.numpy, np.conj(init_sv.numpy))

        # Apply unitary on qubits by tensoring with identities
        # full_circ = Circuit(num_qubits)
        # full_circ.append_gate(ConstantUnitaryGate(rand_un), qubits.tolist())
        # full_un = full_circ.get_unitary()
        full_un = embed_unitary(rand_un, qubits, num_qubits)

        # rho_orig = full_un @ rho_in @ np.conj(full_un.T)
        rho_orig = apply_unitary_normal(full_un, rho_in)

        # Now run with get_single_rho
        rho_out = get_single_rho(rand_un, rho_in, qubits)

        # Now run with superoperator
        superop = np.kron(rand_un.conj(), rand_un)

        rho_out_super = apply_superoperator(rho_in, superop,
                                             num_qubits, qubits)


        # Assert the trace distance is small
        eps = trace_distance(rho_out, rho_orig)
        eps_super = trace_distance(rho_out_super, rho_orig)
        print(f"Test {i}: Trace Distance = {eps}, Superoperator Trace Distance = {eps_super}")

async def test_circ_runner(circ_name: str = "LiH_jw_long", 
                      max_tol: float = 2.0):
    
    good_blocks = calculate_good_blocks(
        circ_name=circ_name,
        checkpoint_form=checkpoint_form,
        max_tol=max_tol,
        cliff_t=CLIFF_T
    )[0]
    # Just use 2 of the small blocks for each large block
    for large_block_num in good_blocks:
        good_blocks[large_block_num] = set(list(good_blocks[large_block_num])[:7])

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
        cliff_t=CLIFF_T
    )

    await runner.initialize()

    init_sv = StateVector.random(num_qubits)
    rho_in = np.outer(init_sv.numpy, np.conj(init_sv.numpy))
    # Get original unitary for comparison
    original_unitary = orig_circ.get_unitary()
    rho_orig = original_unitary @ rho_in @ np.conj(original_unitary.T)
    rho_out = runner.run(rho_in, qubits=np.arange(num_qubits))
    # Calculate trace distance of rhos
    ens_eps = trace_distance(rho_out, rho_orig)
    print("Ensemble Trace Distance: ", ens_eps)


async def test_large_runner(circ_name: str = "LiH_jw_long", 
                      large_block_num: str = "0", 
                      max_tol: float = 2.0):

    good_blocks = calculate_good_blocks(
        circ_name=circ_name,
        checkpoint_form=checkpoint_form,
        max_tol=max_tol,
        cliff_t=CLIFF_T
    )[0]

    good_blocks[large_block_num] = ['0', '6', '7']

    updated_data = update_partitioned_data(
        partitioned_data=all_partitioned_data[circ_name],
        good_blocks=good_blocks
    )

    print("Updated Data Keys: ", list(updated_data[large_block_num][0].keys()))

    orig_circ: Circuit = all_partitioned_data[circ_name][large_block_num][1]
    num_qubits = orig_circ.num_qudits

    runner = create_large_block_runner(
        circ_name=circ_name,
        large_block_num=large_block_num,
        max_tol=max_tol,
        partitioned_data=updated_data[large_block_num],
        checkpoint_form=checkpoint_form,
        cliff_t=CLIFF_T
    )

    await runner.initialize("test_large_runner")
    runner.save("test_large_runner")

    init_sv = StateVector.random(orig_circ.num_qudits)
    rho_in = np.outer(init_sv.numpy, np.conj(init_sv.numpy))

    # Get original unitary for comparison
    original_circ = updated_data[large_block_num][1]
    original_unitary = original_circ.get_unitary()
    rho_orig = original_unitary @ rho_in @ np.conj(original_unitary.T)

    rho_out = runner.run(rho_in, qubits=np.arange(num_qubits))

    # Calculate trace distance of rhos
    # eps = trace_distance(sample_rho_out, rho_orig)
    ens_eps = trace_distance(rho_out, rho_orig)

    print("Ensemble Trace Distance: ", ens_eps)



async def test_small_runner(circ_name: str = "heisenberg7", 
                      large_block_num: str = "0", 
                      max_tol: float = 2.0):

    sub_block_num = "00"
    ensemble_file_names = get_file_names(
        large_checkpoint_dir=checkpoint_form.format(
            circ_name=circ_name,
            large_block_num=large_block_num,
            max_tol=max_tol
        ),
        small_block_num=sub_block_num
    )[:-1]

    start = time.time()


    runner = DensityMatrixRunner(
        qubits=[0,1,2,3],
        ensemble_file_names=ensemble_file_names,
        cliff_t=CLIFF_T
    )

    init_sv = StateVector.random(4)
    rho_in = np.outer(init_sv.numpy, np.conj(init_sv.numpy))

    # Get original unitary for comparison
    original_circ = all_partitioned_data[circ_name][large_block_num][0][sub_block_num]
    original_unitary = original_circ.get_unitary()
    rho_orig = original_unitary @ rho_in @ np.conj(original_unitary.T)


    # # Get a random unitary from runner sampler to compare against
    # u, _ = next(runner.sampler)
    # sample_rho_out = u @ rho_in @ np.conj(u.T)
    # runner.sampler.reset()

    start = time.time()
    await runner.initialize()

    initialization_time = time.time() - start
    print(f"Initialization Time: {initialization_time}")

    start = time.time()
    rho_out = runner.run(rho_in)
    run_time = time.time() - start
    print(f"Run Time: {run_time}")



    # Calculate trace distance of rhos
    # eps = trace_distance(sample_rho_out, rho_orig)
    ens_eps = trace_distance(rho_out, rho_orig)

    # print("Final Time: ", final_time)
    # print("Trace Distance: ", eps)
    print("Ensemble Trace Distance: ", ens_eps)

if __name__ == '__main__':
    # test_small_runner()
    # test_large_runner(circ_name="mult8")

    compiler = Compiler(num_workers=256)
    circ = Circuit(2)

    start = time.time()
    compiler.compile(circuit=circ, workflow=[TestRunner()])
    print("Total Time: ", time.time() - start)