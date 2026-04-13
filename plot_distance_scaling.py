
from sys import argv
from util.plot_lib import plot_dm_data

import matplotlib.pyplot as plt

if __name__ == '__main__':
    TRACE_DISTANCE = bool(int(argv[1])) if len(argv) > 1 else False

    if TRACE_DISTANCE:
        circ_names = [
            # "qae11",
            "qaoa10",
            "FermiHubbard2x2_jw_long",
            "heisenberg7",
            "LiH_jw_long",
        ]
    else:
        circ_names = [
            "FermiHubbard2x2_jw_long",
            "heisenberg7",
            "LiH_jw_long",
        ]
        
    cliff_t = True
    cliff_t_string = "_clifft" if cliff_t else ""
    folder_form = "ensemble_dms_{circ_name}" + cliff_t_string + "_final{extra}/"

    fig, axs = plt.subplots(1, 1, figsize=(10, 6))

    if TRACE_DISTANCE:
        y_label = "Trace Distance of Channel"
    else:
        y_label = "Hamiltonian Observable Error of Channel"

    plot_dm_data(circ_names, axs, folder_form=folder_form, y_label=y_label, 
                 calc_obs=not TRACE_DISTANCE, cliff_t=cliff_t, 
                 extras=["_N", "_Sz"])

    if TRACE_DISTANCE:
        plt.savefig('trace_distance_scaling_clifft_final.png', dpi=300)
    else:
        plt.savefig('other_hamiltonian_scaling_clifft_final.png', dpi=300)