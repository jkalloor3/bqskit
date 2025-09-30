import os
from qiskit import QuantumCircuit
from qiskit.visualization import circuit_drawer
from sys import argv
import numpy as np
from pathlib import Path

# Go through all folders in fixed_block_checkpoint_min. For every qasm, read it in
# and draw the circuit. Save the circuit as a png in the same folder.
def draw_all_blocks(base_dir, output_dir):
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.qasm'):
                qasm_path = os.path.join(root, file)
                circuit = QuantumCircuit.from_qasm_file(qasm_path)
                output_path = os.path.join(output_dir, file.replace('.qasm', '.png'))
                try:
                    circuit_drawer(circuit, output='mpl', filename=output_path)
                    print(f"Saved {output_path}")
                except:
                    pass

def draw_random_ensemble_blocks(input_folder: str):
    # Walk through input folder
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith('.qasms'):
                qasm_path = os.path.join(root, file)
                ens_ind = file.split(".")[0].split("_")[-2]
                images_folder = os.path.join(root, f"images_{ens_ind}")
                Path(images_folder).mkdir(parents=True, exist_ok=True)
                print("Saving Images to ", images_folder, flush=True)
                with open(qasm_path, "r") as f:
                    qasms = f.read().split("\nBREAK\n")

                # Randomly pick 5 qasms
                rand_inds = np.random.choice(len(qasms), min(5, len(qasms)), replace=False)
                qasms = [qasms[i] for i in rand_inds]
                circuits = [QuantumCircuit.from_qasm_str(qasm) for qasm in qasms]
                for i, circ in enumerate(circuits):
                    output_path = os.path.join(images_folder, f"circ_{i}.png")
                    try:
                        circuit_drawer(circ, output='mpl', filename=output_path)
                    except:
                        pass


if __name__ == "__main__":
    # checkpoint_dir = "good_blocks_tket"
    # output_dir = "good_blocks_tket_pngs"
    checkpoint_dir = "block_checkpoints_final_paper_tket"
    # checkpoint_dir = "block_checkpoints_final_paper_tket/adder9_0_0.8"
    print(f"Drawing circuits in {checkpoint_dir}")
    draw_random_ensemble_blocks(checkpoint_dir)
    # draw_all_blocks(checkpoint_dir, output_dir)

