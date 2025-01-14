import os
import json

import matplotlib.pyplot as plt

def plot_data(data: dict, axes: plt.Axes):
    x_axis = data["Ensemble Size"]

    headers = ["Trace Distance", "TVD", "Frobenius Distance"]
    for header in headers:
        if header not in data:
            print(f"Header {header} not found in data")
            continue
        y = data[header]
        axes.plot(x_axis, y, label=header)
        axes.plot(x_axis, y, '*')

    # Set y to log scale
    axes.set_yscale('log')
    axes.legend()



# Path to the folder containing JSON files
# folder_path = '/pscratch/sd/j/jkalloor/bqskit/no_qp_conv_data'
# folder_path = '/pscratch/sd/j/jkalloor/bqskit/qp_conv_data'
folder_path = '/pscratch/sd/j/jkalloor/bqskit/no_qp_conv_data_noisy_new'

# Initialize lists to store data
ensemble_sizes = []
bias_reductions = []

# Iterate over all files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.json'):
        file_path = os.path.join(folder_path, filename)

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Read JSON file
        with open(file_path, 'r') as file:
            data = json.load(file)
            plot_data(data, ax)

        title = filename.split('.')[0]
        fig.savefig(f'conv_{title}_qp_noisy.png')