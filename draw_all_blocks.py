import os
from qiskit import QuantumCircuit
from qiskit.visualization import circuit_drawer
from sys import argv

# Go through all folders in fixed_block_checkpoint_min. For every qasm, read it in
# and draw the circuit. Save the circuit as a png in the same folder.
def draw_all_blocks(base_dir):
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.qasm'):
                qasm_path = os.path.join(root, file)
                circuit = QuantumCircuit.from_qasm_file(qasm_path)
                output_path = os.path.join(root, file.replace('.qasm', '.png'))
                circuit_drawer(circuit, output='mpl', filename=output_path)
                print(f"Saved {output_path}")

if __name__ == "__main__":
    circ_name = argv[1]
    timestep = int(argv[2])
    tol = int(argv[3])
    num_unique_circs = int(argv[4])
    opt = bool(int(argv[5])) if len(argv) > 5 else False
    opt_str = "_opt" if opt else ""
    big_block_size = 8
    small_block_size = 3
    checkpoint_dir = f"fixed_block_checkpoints_min{opt_str}/{circ_name}_{timestep}_{tol}_{big_block_size}_{small_block_size}/"
    print(f"Drawing circuits in {checkpoint_dir}")
    draw_all_blocks(checkpoint_dir)
