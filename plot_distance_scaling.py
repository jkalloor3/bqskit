
from sys import argv
from util.plot_lib import plot_dm_data

import matplotlib.pyplot as plt

if __name__ == '__main__':
    TRACE_DISTANCE = bool(int(argv[1])) if len(argv) > 1 else False

    if TRACE_DISTANCE:
        circ_names = [
            "qaoa10",
            "qpe_11",
            "mult8",
            "draper_adder_12",
            "qae11",
        ]
    else:
        circ_names = [
            # "lgt_11",
            # "QITE_8_0",
            "FermiHubbard2x2_jw_long",
            "heisenberg7",
            "LiH_jw_long",
            "neutrino_NX_3_NF_2_jw_long"
        ]
        

    folder_form = "ensemble_dms_{circ_name}/"

    fig, axs = plt.subplots(1, 1, figsize=(10, 6))

    if TRACE_DISTANCE:
        y_label = "Trace Distance of Channel"
    else:
        y_label = "Hamiltonian Observable Error of Channel"

    plot_dm_data(circ_names, axs, folder_form=folder_form, y_label=y_label, diff=False)

    if TRACE_DISTANCE:
        plt.savefig('trace_distance_scaling.png', dpi=300)
    else:
        plt.savefig('hamiltonian_scaling.png', dpi=300)