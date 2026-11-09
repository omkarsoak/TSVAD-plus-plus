import os
import shutil

# Define the base path
import sys
base_path = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("SIM_DATA", "/path/to/simu3/data")  # usage: python 3.2-move_to_all_files.py <simu3/data>

# Define the target directory
target_dir = os.path.join(base_path, "all_files")

# Define the subdirectories to move
subdirs = ["rttms", "target_audio", "target_embedding"]

# create subdirs in target_dir if not exist
for subdir in subdirs:
    subdir_path = os.path.join(target_dir, subdir)
    if not os.path.exists(subdir_path):
        os.makedirs(subdir_path)

# Initialize a dictionary to keep track of the number of files moved
file_counts = {subdir: 0 for subdir in subdirs}

# Iterate over all directories in the base path
for dir_name in os.listdir(base_path):
    dir_path = os.path.join(base_path, dir_name)
    
    # Skip the target directory
    if dir_path == target_dir:
        continue
    
    # If the directory contains the subdirectories, move them
    if all(os.path.isdir(os.path.join(dir_path, subdir)) for subdir in subdirs):
        for subdir in subdirs:
            subdir_path = os.path.join(dir_path, subdir)
            
            # Move all files in the subdirectory to the corresponding directory in the target directory
            for file_name in os.listdir(subdir_path):
                file_path = os.path.join(subdir_path, file_name)
                shutil.move(file_path, os.path.join(target_dir, subdir, file_name))
                
                # Increment the count for this subdirectory
                file_counts[subdir] += 1

# Print the total number of files moved for each subdirectory
for subdir, count in file_counts.items():
    print(f"Moved {count} files from {subdir} directories.")
