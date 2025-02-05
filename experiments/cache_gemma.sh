#!/bin/bash

# Define the arguments as an array of parameter sets
declare -a args=(
    # Format: "model_size width l0 layer"
    "2b 65k 116 18"
    "9b 131k 98 28"
    # "27b 131k 72 34"
)

# Iterate through each parameter set
for params in "${args[@]}"; do
    # Split the parameter string into individual variables
    read -r model_size width l0 layer <<< "$params"
    
    echo "Running with model_size=$model_size width=$width l0=$l0 layer=$layer"
    
    python experiments/cache_gemma.py \
        --model-size "$model_size" \
        --width "$width" \
        --l0 "$l0" \
        --layer "$layer"
done