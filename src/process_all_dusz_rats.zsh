#!/usr/bin/env zsh

# Usage: ./process_rats.zsh /path/to/folder 2
#  - /path/to/folder = the base folder name
#  - 2 = dataset number

folder_name="$1"
dataset_number="$2"

# Adjust the path structure as needed
base_dir="${folder_name}/Dataset_${dataset_number}"

# Iterate over each subdirectory in base_dir
for rat_dir in "$base_dir"/*; do
    if [ -d "$rat_dir" ]; then
        # Extract just the rat's directory name (e.g. "A3701-191119")
        rat_id=$(basename "$rat_dir")
        echo "Processing $rat_id ..."
        python src/dusz_ratemaps_xy_hd.py --dataset "$dataset_number" --rat "$rat_id" --n_hd_bins 10 --n_x_bins 10 --n_y_bins 10
    fi
done

