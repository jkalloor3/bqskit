from bqskit.ir.circuit import Circuit
from bqskit.ir.gates import CircuitGate
from sys import argv
import os


import pickle

from bqskit.compiler import Compiler
from bqskit.passes import ScanPartitioner, ExtendBlockSizePass


from common.io import (load_block, get_block_names)
from util.common import load_circuit
from util.unitary_dm_pass import DMEvaluator, get_sub_block_nums
from util.hamiltonian import generate_hamiltonian, generate_init_state

# Get partitioned circuits for all 8-qubit blocks
def partition_circs(compiler: Compiler,
                    circ_name, 
                    block_num: str) -> tuple[dict[str, Circuit], Circuit]:
    workflow = [
        ScanPartitioner(4),
        # ExtendBlockSizePass(4),
    ]
    circ_file = load_block(circ_name, block_num, extra="_tket")
    circ = Circuit.from_file(circ_file)
    return compiler.submit(circ, workflow=workflow)


if __name__ == "__main__":
    circ_names = ["heisenberg7", "qaoa10"]
    # circ_names += ["FermiHubbard2x2_jw_long"]
    compiler = Compiler(num_workers=128)

    all_partitioned_ids = {}
    all_partitioned_data = {}

    cliff_t = False
    cliff_t_string = "_clifft" if cliff_t else ""
    if cliff_t:
        print("Using cliff-t circuits", flush=True)
        base_checkpoint_dir = "small_block_checkpoints_final_paper_4_clifft_tket"
    else:
        print("Using non-cliff-t circuits", flush=True)
        base_checkpoint_dir = "small_block_checkpoints_final_paper_4_more_cx_tket"

    partitioned_data_file = f"partitioned_data_all_circs{cliff_t_string}.pickle"

    missing_circ_names = circ_names.copy()
    if os.path.exists(partitioned_data_file):
        with open(partitioned_data_file, "rb") as f:
            all_partitioned_data = pickle.load(f)
        print("Loaded partitioned data from file.", list(all_partitioned_data.keys()))
        # Only run on circs that are not in data
        missing_circ_names = [name for name in circ_names if name not in all_partitioned_data]
    
    print(f"Missing circ names: {missing_circ_names}", flush=True)
    for circ_name in missing_circ_names:
        all_partitioned_ids[circ_name] = {}
        for large_block_num in get_block_names(circ_name, extra="_tket"):
            id = partition_circs(compiler, circ_name, large_block_num)
            
            all_partitioned_ids[circ_name][large_block_num] = id
        
        all_partitioned_data[circ_name] = {}
        checkpoint_folder_form = f"{base_checkpoint_dir}/{circ_name}" + "_{large_block_num}_" + f"*/"
        for large_block_num in get_block_names(circ_name, extra="_tket"):
            out_circ = compiler.result(all_partitioned_ids[circ_name][large_block_num])
            base_dir = checkpoint_folder_form.format(large_block_num=large_block_num)
            sub_block_names = get_sub_block_nums(base_dir)
            assert out_circ.num_operations == len(sub_block_names), f"Number of operations in circuit {circ_name} does not match number of sub-blocks: {out_circ.num_operations} != {len(sub_block_names)}"
            sub_block_circs = {}
            for i, op in enumerate(out_circ.operations()):
                assert isinstance(op.gate, CircuitGate)
                block_name = sub_block_names[i]
                sub_block_circs[block_name] = op.gate._circuit
            all_partitioned_data[circ_name][large_block_num] = (sub_block_circs, out_circ)
    
    with open(partitioned_data_file, "wb") as f:
        pickle.dump(all_partitioned_data, f)

    exit(0)

    compiler_ids = []

    print("Circ names to process:", circ_names, flush=True)

    checkpoint_folder_form = (base_checkpoint_dir + 
                              "/{circ_name}_{large_block_num}_{max_tol}/")
    for circ_name in circ_names:
        full_circ = load_circuit(circ_name)
        full_circ.remove_all_measurements()
        ham = None
        init_sv = None
        # ham = generate_hamiltonian(circ_name, full_circ.num_qudits)
        # init_sv = generate_init_state(circ_name, full_circ.num_qudits)
        for tol in [1.0, 2.0, 3.0, 4.0, 5.0]:
            workflow = [
                DMEvaluator(
                    circ_name=circ_name,
                    max_tol=tol,
                    partitioned_data=all_partitioned_data[circ_name],
                    checkpoint_form=checkpoint_folder_form,
                    ham=ham,
                    partitioned_circ_file=f"partitioned_circs/{circ_name}.pickle",
                    save_dir=f"ensemble_dms_{circ_name}{cliff_t_string}_final/",
                    cliff_t=cliff_t,
                    init_sv=init_sv,
                    run_blocks=True
                )
            ]
            # Await the result before starting a new one
            compiler.compile(full_circ, workflow=workflow)

    for id in compiler_ids:
        compiler.result(id)

    compiler.close()