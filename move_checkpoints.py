import os
import shutil
import pandas as pd
import glob

base_dir = '/pscratch/sd/j/jkalloor/bqskit/block_checkpoints_final_paper'

# Ensure the destination directory exists
# os.makedirs(destination_dir, exist_ok=True)

def move_folder(folder_path, destination_path):
    # Delete all block files
    block_file = os.path.join(folder_path, 'block_*.*')
    block_files = glob.glob(block_file)
    for file in block_files:
        os.remove(file)
    shutil.move(folder_path, destination_path)

def move_and_delete_folder(folder_path, destination_path):
    # Move all files from folder_path to destination_path
    for filename in os.listdir(folder_path):
        shutil.move(os.path.join(folder_path, filename), destination_path)
    # Delete the folder
    shutil.rmtree(folder_path)

# for dir in os.listdir(source_dir):
#     csv_path = os.path.join(source_dir, dir, '*.csv')
#     files = glob.glob(csv_path)
#     if len(files) == 0:
#         continue
#     print(files)
#     df = pd.read_csv(files[0])
#     if 'Ratio' in df.columns:
#         min_ratio = df['Ratio'].min()
#         if min_ratio < 200:
#             folder_to_move = os.path.join(source_dir, dir)
#             # Remove _250
#             parts = dir.split('_')
#             if int(parts[-1]) == 250:
#                 parts = parts[:-1]
#             else:
#                 # Don't copy over
#                 continue
#             new_folder_name = "_".join(parts)
#             destination_path = os.path.join(destination_dir, new_folder_name)
#             move_folder(folder_to_move, destination_path)
#             print(f"Moved {folder_to_move} to {destination_path}")

for dir in os.listdir(base_dir):
    bad_dir = os.path.join(base_dir, dir, dir)

    if os.path.isdir(bad_dir):
        correct_dir = os.path.join(base_dir, dir)
        print(f"Moving {bad_dir} to {correct_dir}")
        move_and_delete_folder(bad_dir, correct_dir)
