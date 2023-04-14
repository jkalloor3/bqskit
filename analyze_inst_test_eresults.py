#%%

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from inst_test_utils import  SlurmSubmissionsDb


submission_db_name = 'inst_test1.sqlite'
sdb = SlurmSubmissionsDb(db_name=submission_db_name)

#%%

df = pd.DataFrame(sdb.get_all_submissions())
df['qubit_count'] = df['qubit_count'].astype(int)
# %%

# calculate total succus rate
print(df.groupby('inst_name')['inst_succsed'].mean())
# %%
print(df.groupby(['inst_name', 'inst_succsed'])['inst_time'].mean().unstack())
# %%


# group the data by circ_name, inst_name, and qubit_count, and calculate the mean of inst_time
avg_time = df.groupby(['orig_circ_name', 'inst_name', 'qubit_count'])['inst_time'].mean()

# create a new data frame with the result of the groupby operation
df_avg_time = avg_time.reset_index()

# loop over the unique circ_names
for circ_name in df_avg_time['orig_circ_name'].unique():
    # select the subset of data for the current circ_name
    subset = df_avg_time[df_avg_time['orig_circ_name'] == circ_name]
    
    # create a new plot for the current circ_name
    plt.figure()
    
    # loop over the unique combinations of inst_name for the current circ_name
    for inst_name in subset['inst_name'].unique():
        # select the subset of data for the current combination of circ_name and inst_name
        inst_subset = subset[subset['inst_name'] == inst_name]
        
        # plot the subset of data as a line plot with markers
        plt.plot(inst_subset['qubit_count'], inst_subset['inst_time'], label=f"{inst_name}", marker='o')
    
    # add labels and legend to the plot
    plt.xlabel('qubit_count')
    plt.ylabel('Average inst_time')
    plt.title(circ_name)
    plt.legend()

    # set x-axis to integers
    plt.xticks(range(min(subset['qubit_count']), max(subset['qubit_count'])+1))
    
    
    # set y-axis to log scale
    plt.yscale('log')

    plt.savefig(circ_name + '_runtime.pdf', bbox_inches='tight')
    
# show all the plots
plt.show()

# %%

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt


# # Filter out the rows where inst_succsed is NaN
# # Group the data by orig_circ_name, inst_name, and qubit_count
# grouped = df[df["inst_succsed"].notna()].groupby(["orig_circ_name", "inst_name", "qubit_count"])



# # Iterate over each orig_circ_name and create a separate plot for each one
# for orig_circ_name, group in grouped:
#     # Set up the plot
#     fig, ax = plt.subplots(figsize=(10, 8))
#     ax.set_xlabel("Qubit Count", fontsize=14)
#     ax.set_ylabel("Success Rate", fontsize=14)
#     ax.set_title(f"Success Rate vs Qubit Count for {orig_circ_name}", fontsize=16)

#     # Iterate over each inst_name and plot the data
#     for inst_name, inst_group in group.groupby("inst_name"):
#         # Calculate the mean success rate for each qubit count
#         # success_rate = inst_group["inst_succsed"].value_counts(normalize=True).loc[:, True]
#         success_rate = inst_group["inst_succsed"].value_counts(normalize=True).loc[True]

#         success_rate_mean = success_rate.groupby("qubit_count").mean()

#         # Convert the index to integers
#         success_rate_mean.index = success_rate_mean.index.astype(int)

#         # Plot the data as a scatter plot with a line connecting the points
#         ax.plot(success_rate_mean.index, success_rate_mean, "-o", label=inst_name)

#     # Add a legend and show the plot
#     ax.legend()
#     plt.show()


#%%
import matplotlib.pyplot as plt
import pandas as pd


# Compute success rate
df_success = df[df["inst_succsed"] == True]
success_rate = df_success.groupby(["orig_circ_name", "inst_name", "qubit_count"]).size() / df.groupby(["orig_circ_name", "inst_name", "qubit_count"]).size()
success_rate = success_rate.reset_index(name="success_rate")


num_insts = df['inst_name'].nunique()

# Loop through each unique INST for the current FILE_NAME
colors = plt.cm.tab10.colors[:num_insts]

# Set the width of each bar based on the number of unique INSTs
bar_width = 0.8 / num_insts

# Create a separate plot for each orig_circ_name
for orig_circ_name, df_orig in success_rate.groupby("orig_circ_name"):
    # Create a separate subplot for each inst_name
    fig, ax = plt.subplots()


    # for inst_name, df_inst in df_orig.groupby("inst_name"):
    for i, inst in enumerate(df_orig['inst_name'].unique()):
        df_inst = df_orig[df_orig['inst_name'] == inst]
        # Plot the success rate for each inst_name
        # Calculate the x-coordinates for the current INST's bars
        x_pos = np.arange(min(df_inst['qubit_count']) + i * bar_width - (bar_width * (num_insts - 1) / 2) , max(df_inst['qubit_count']) + 0.5 + i * bar_width )

        ax.bar(x_pos,  df_inst["success_rate"], width=bar_width, label=inst, color=plt.cm.tab10(i), hatch='/', edgecolor='black')


    # Set the axis labels and title
    ax.set_xlabel("qubit_count")
    ax.set_ylabel("Success rate")
    ax.set_title(f"Success rate by inst_name for {orig_circ_name}")
    ax.set_ylim([0, 1.4])
    ax.set_xticks(df_orig["qubit_count"].unique())
    ax.legend()

    plt.savefig(orig_circ_name + '_success_rate.pdf', bbox_inches='tight')
plt.show()

# %%
