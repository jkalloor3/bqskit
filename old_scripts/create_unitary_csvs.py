import os
import csv
import pickle
from typing import Any
# from bqskit.qis import UnitaryMatrix
# from bqskit.ir import Circuit

def extract_unitaries_to_csv(folder_path, output_csv_base):
    # Iterate through each file in the folder
    for filename in os.listdir(folder_path):
        if filename.startswith('block_') and filename.endswith('.data'):
            file_path = os.path.join(folder_path, filename)
            
            # Read the data file
            with open(file_path, 'rb') as file:
                data = pickle.load(file)

            # Extract the target unitaries
            unitaries: list = data.get("ensemble_targets", [])

            circ_dists: list[list[tuple[Any, float]]] = data.get("organized_scan_sols", [])
            
            output_csv = output_csv_base + filename.replace('.data', '.csv')

            # Open the CSV file for writing
            with open(output_csv, mode='w', newline='') as csv_file:
                csv_writer = csv.writer(csv_file)
                
                # Write the header
                # csv_writer.writerow(['unitary'])
                
                # Write each target unitary, followed by circuit unitaries
                for i, unitary in enumerate(unitaries):
                    csv_writer.writerow(unitary.numpy.flatten())
                    # Write all circuit unitaries
                    circuits = [circuit for circuit, _ in circ_dists[i]]
                    for i in range(2):
                        if len(circuits) > i:
                            circuit = circuits[i]
                            csv_writer.writerow(circuit.get_unitary().numpy.flatten())
                        else:
                            # Write a blank line
                            csv_writer.writerow(['' for _ in range(2**6)])

if __name__ == '__main__':
    # Example usage
    # folder_path = 'hamiltonian_perturbation_checkpoints_leap/adder9_hard_1e-05_1.0000000000000002e-06_72'
    folder_path = "hamiltonian_perturbation_checkpoints_leap/adder9_hard_1e-05_1e-07_72"
    output_csv = 'adder9_hard_set_2_'
    extract_unitaries_to_csv(folder_path, output_csv)