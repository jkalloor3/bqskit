#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


filename = './del_gates_4_3.csv'
filename = './del_gates_4_5.csv'
filename = './del_gates_4_7.csv'
filename = './del_gates_4_10.csv'


orig_circuits_props = {
                'hub4.qasm'       :{'1Q_COUNT': 155,  '2Q_COUNT':180},
                'qaoa5.qasm'      :{'1Q_COUNT': 27,   '2Q_COUNT':42},
                'grover5_u3.qasm' :{'1Q_COUNT': 80,   '2Q_COUNT':48},
                'adder9_u3.qasm'  :{'1Q_COUNT': 64,   '2Q_COUNT':98},
                'hub18.qasm'      :{'1Q_COUNT': 1992, '2Q_COUNT':3541},
                'adder63_u3.qasm' :{'1Q_COUNT': 2885, '2Q_COUNT':1405},
                'shor26.qasm'     :{'1Q_COUNT': 20896,'2Q_COUNT':21072},

                'add17.qasm'     :{'1Q_COUNT': 348,'2Q_COUNT':232},
                'heisenberg7.qasm':{'1Q_COUNT': 490,'2Q_COUNT':360},
                'heisenberg8.qasm':{'1Q_COUNT': 570,'2Q_COUNT':420},
                'heisenberg64.qasm':{'1Q_COUNT': 5050,'2Q_COUNT':3780},
                'hhl8.qasm'     :{'1Q_COUNT': 3288,'2Q_COUNT':2421},
                'mult8.qasm'     :{'1Q_COUNT': 210,'2Q_COUNT':188},
                'mult16.qasm'     :{'1Q_COUNT': 1264,'2Q_COUNT':1128},
                'mult64.qasm'     :{'1Q_COUNT': 61600,'2Q_COUNT':54816},
                'qae13.qasm'     :{'1Q_COUNT': 247,'2Q_COUNT':156},
                'qae11.qasm'     :{'1Q_COUNT': 176,'2Q_COUNT':110},
                'qae33.qasm'     :{'1Q_COUNT': 1617,'2Q_COUNT':1056},
                'qae81.qasm'     :{'1Q_COUNT': 7341,'2Q_COUNT':4840},
                'qpe8.qasm'     :{'1Q_COUNT': 519,'2Q_COUNT':372},
                'qpe10.qasm'     :{'1Q_COUNT': 1681,'2Q_COUNT':1260},
                'qpe12.qasm'     :{'1Q_COUNT': 3582,'2Q_COUNT':2550},
                'tfim16.qasm'     :{'1Q_COUNT': 916,'2Q_COUNT':600},
                'tfim8.qasm'     :{'1Q_COUNT': 428,'2Q_COUNT':280},
                'vqe14.qasm'     :{'1Q_COUNT': 10792,'2Q_COUNT':20392},
                'vqe12.qasm'     :{'1Q_COUNT': 4157,'2Q_COUNT':7640},
                'vqe5.qasm'     :{'1Q_COUNT': 132,'2Q_COUNT':91},
                'tfim400.qasm'     :{'1Q_COUNT': 88235,'2Q_COUNT':87670},






                    }



#%%



# Read in the CSV file using pandas, and specify the column names manually
column_names = ['True', 'INST', 'FILE_NAME', 'MULTISTARTS', 'PARTITION_SIZE', 'QUBIT_COUNT', 'RUNTIME', '1Q_COUNT', '2Q_COUNT', 'NODES', 'WORKERS_PER_NODE', 'GPUS']
data = pd.read_csv(filename, names=column_names)


for q in [1,2]:
    data[f'{q}Q_REDUCTION'] = None
    for index, row in data.iterrows():
        file_name = row['FILE_NAME']    
        base_line = orig_circuits_props[file_name][f'{q}Q_COUNT']

        reduction = 100 * (base_line - row[f'{q}Q_COUNT'])/base_line
        data.at[index, f'{q}Q_REDUCTION'] = reduction
    data[f'{q}Q_REDUCTION'] = data[f'{q}Q_REDUCTION'].astype('float64')

# Group the data by FILE_NAME, INST, and PARTITION_SIZE, and compute the mean of the RUNTIME column
grouped_data = data.groupby(['FILE_NAME', 'INST', 'PARTITION_SIZE'], as_index=False)['RUNTIME', '2Q_COUNT', '1Q_COUNT', '1Q_REDUCTION', '2Q_REDUCTION'].mean()
#%%
# Loop through each unique FILE_NAME
for file_name in grouped_data['FILE_NAME'].unique():
    if file_name in ['heisenberg8.qasm']:
        continue
    # Subset the data for the current FILE_NAME
    file_data = grouped_data[grouped_data['FILE_NAME'] == file_name]
    circuit_name = file_name.split('.')[0]

    fig, ax = plt.subplots()

    # Loop through each unique INST for the current FILE_NAME
    for inst in file_data['INST'].unique():
        # Subset the data for the current INST
        inst_data = file_data[file_data['INST'] == inst]

        # Create a line plot for the current INST
        ax.plot(inst_data['PARTITION_SIZE'], inst_data['RUNTIME'], label=inst, marker='o', markersize=6)

    # Set the title and axis labels for the current plot
    ax.set_title(f'{circuit_name} Compile Time')
    ax.set_xlabel('Partition Size')
    ax.set_ylabel('Compile Time [s] (log scale)')

    # Set the y-axis to a logarithmic scale
    ax.set_yscale('log')
    ax.set_ylim(1)

    # Set the x-axis ticks to integers
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # Add a legend to the plot
    ax.legend()

    # Show the plot
    # plt.show()
    plt.savefig(circuit_name + '_run_time.pdf', bbox_inches='tight')

    num_insts = file_data['INST'].nunique()

    # Loop through each unique INST for the current FILE_NAME
    colors = plt.cm.tab10.colors[:num_insts]

    # Set the width of each bar based on the number of unique INSTs
    bar_width = 0.8 / num_insts

    for q in [1,2]:
        fig, ax = plt.subplots()
        bars_l = []
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

        max_bar_value = 0

        for i, inst in enumerate(file_data['INST'].unique()):
            # Subset the data for the current INST
            inst_data = file_data[file_data['INST'] == inst]

            # Calculate the x-coordinates for the current INST's bars
            x_pos = np.arange(min(inst_data['PARTITION_SIZE']) + i * bar_width - (bar_width * (num_insts - 1) / 2) , max(inst_data['PARTITION_SIZE']) + 0.5 + i * bar_width )
            
            max_bar_value = max(max_bar_value, max(inst_data[f'{q}Q_REDUCTION']) )
            bars_l.append(ax.bar(x_pos, inst_data[f'{q}Q_REDUCTION'], width=bar_width, label=inst, color=plt.cm.tab10(i), hatch='/', edgecolor='black'))

    
        
        offset = max_bar_value / 100
        if q==1:
            ax.set_ylabel(f'U3 Gate Reduction [%]')
            
        else:
            # ax.set_ylim(0, 10)
            ax.set_ylabel(f'CNOT Gate Reduction [%]')
            # offset = max(0.5, max_bar_value / 100)

        for bars in bars_l:
            for bar in bars:
                h = bar.get_height()
                if inst_data['PARTITION_SIZE'].nunique() > 3:
                    r = int(h)
                elif h ==0:
                    r = 0
                elif h < 0.1:
                    r = f"{h:.0e}"
                else:
                    r = round(h,1)

                ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + offset,
                    r,
                    horizontalalignment='center',
                    fontsize=7,
                    # weight='bold',
                    )


        ax.set_xlabel('Partition Size')



        ax.legend()
        

  
        if q==1:
            ax.set_title(f'U3 Reduction in {circuit_name}')
            plt.savefig(circuit_name + '_u3_reduction.pdf', bbox_inches='tight')
        else:
            ax.set_title(f'CNOT Reduction in {circuit_name}')
            plt.savefig(circuit_name + '_cnot_reduction.pdf', bbox_inches='tight')


# %%
# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np

# # Read in the CSV file using pandas, and specify the column names manually
# column_names = ['True', 'INST', 'FILE_NAME', 'MULTISTARTS', 'PARTITION_SIZE', 'QUBIT_COUNT', 'RUNTIME', '1Q_COUNT', '2Q_COUNT', 'NODES', 'WORKERS_PER_NODE', 'GPUS']
# data = pd.read_csv('./del_gates_4_3.csv', names=column_names)

# # Group the data by FILE_NAME, INST, and PARTITION_SIZE, and compute the mean of the RUNTIME column
# grouped_data = data.groupby(['FILE_NAME', 'INST', 'PARTITION_SIZE'], as_index=False)['RUNTIME', '2Q_COUNT', '1Q_COUNT'].mean()

# # Loop through each unique FILE_NAME
# for file_name in grouped_data['FILE_NAME'].unique():
#     # Subset the data for the current FILE_NAME
#     file_data = grouped_data[grouped_data['FILE_NAME'] == file_name]

#     # Create a new figure and axis for the current FILE_NAME
#     fig, ax1 = plt.subplots()

    

#     num_insts = len(file_data['INST'].unique())

#     # Loop through each unique INST for the current FILE_NAME
#     colors = plt.cm.tab10.colors[:num_insts]

#     # Set the width of each bar based on the number of unique INSTs
#     bar_width = 0.8 / num_insts


#     for i, inst in enumerate(file_data['INST'].unique()):
#         # Subset the data for the current INST
#         inst_data = file_data[file_data['INST'] == inst]

#         # Calculate the x-coordinates for the current INST's bars
#         x_pos = np.arange(min(inst_data['PARTITION_SIZE']) + i * bar_width - (bar_width * (num_insts - 1) / 2) , max(inst_data['PARTITION_SIZE']) + 0.5 + i * bar_width )
        

#         # Create a bar plot for the 2Q_COUNT data
#         ax1.bar(x_pos, inst_data['2Q_COUNT'], width=bar_width, label=inst, color=plt.cm.tab10(i))

#         ax1.set_xlabel('PARTITION_SIZE')
#         ax1.set_ylabel('2Q_COUNT')

#         # Set the y-axis label color for the bar plot
#         ax1.tick_params(axis='y')

#     ax1.legend()
#     ax1.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
#     # Show the plot
#     plt.show()

#     fig, ax2 = plt.subplots()

    
    

#     for i, inst in enumerate(file_data['INST'].unique()):
#         # Subset the data for the current INST
#         inst_data = file_data[file_data['INST'] == inst]
#         # Create a line plot for the current INST
#         ax2.plot(inst_data['PARTITION_SIZE'], inst_data['RUNTIME'], label=inst, marker='o', markersize=6)

#     # Set the title and axis labels for the current plot
#     ax2.set_title(file_name)
#     ax2.set_ylabel('RUNTIME (log scale)')

#     # Set the y-axis to a logarithmic scale
#     ax2.set_yscale('log')

#     # Add a legend to the plot
#     ax2.legend()
#     ax2.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
#     plt.show()    

# # %%
