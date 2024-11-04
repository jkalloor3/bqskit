import os
import pandas as pd
from sys import argv


nisq_checkpoint = "fixed_block_checkpoints_min"
cliff_checkpoint = "cliff_t_checkpoints"

basic_path = "/pscratch/sd/j/jkalloor/bqskit/{checkpoint}/{file_name}_0_{tol}_8_3"

def combine_csvs(folder_path):
    combined_df = pd.DataFrame()
    block_id = 0
    while True:
        filename = f"block_{block_id}.csv_subselect2"
        file_path = os.path.join(folder_path, filename)
        # print(file_path)
        if os.path.exists(file_path):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)
            
            # Add a row with the filename
            filename_row = pd.DataFrame([[f"BLOCK: {block_id}"] + [''] * (df.shape[1] - 1)], columns=df.columns)
            block_id += 1
            combined_df = pd.concat([combined_df, filename_row, df], ignore_index=True)
        else:
            break
    
    return combined_df

def combine_tols(file_name: str, is_cliff: bool = True):
    total_df = pd.DataFrame()

    if is_cliff:
        checkpoint = cliff_checkpoint
    else:
        checkpoint = nisq_checkpoint

    for tol_num in [1,2,3]:
        folder_path = basic_path.format(checkpoint=checkpoint, file_name=file_name, tol=tol_num)
        df = combine_csvs(folder_path)
        # Add a row with the Tol NUM
        print(df.shape)
        filename_row = pd.DataFrame([[f"TOLERANCE: {tol_num}"] + [''] * (df.shape[1] - 1)], columns=df.columns)
        total_df = pd.concat([total_df, filename_row, df], ignore_index=True)
    
    return total_df


if __name__ == '__main__':
    circ_name = argv[1]
    is_cliff = bool(int(argv[2])) if len(argv) > 2 else True
    total_df = combine_tols(circ_name, is_cliff)
    cliff_str = "cliff" if is_cliff else "nisq"
    total_df.to_csv(f"{circ_name}_combined_{cliff_str}.csv", index=False)