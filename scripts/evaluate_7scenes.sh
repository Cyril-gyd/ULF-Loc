#!/bin/bash

BASE_DATA="datasets/7scenes"
BASE_RESULT="outputs/7scenes"
CFG_FILE="configs/ulfloc_7scenes.yaml"

DATASET="7scenes"
SCENES=(
    "heads"
    "pumpkin" 
    "office"
    "chess"
    "fire"
    "redkitchen"
    "stairs"
)


mkdir -p "$BASE_RESULT/logs"

ulimit -n 65536
export PYTHONGC=2 

function run_task {
    local scene=$1
    local start_time=$(date +%s)
    
    echo "--------------------------------------------------"
    echo "Starting $scene at $(date)"
    echo "Input: $BASE_DATA/7scenes_reference_models/$scene"
    echo "Output: $BASE_RESULT/$scene"
    

    if [ ! -d "$BASE_DATA/7scenes_reference_models/$scene" ]; then
        echo "ERROR: Input directory not found for $scene!"
        return 1
    fi
    
    python ulfloc.py \
        -s "$BASE_DATA/7scenes_reference_models/$scene" \
        -m "$BASE_RESULT/$scene" \
        --data_device cpu \
        --images "../../$scene" \
        --cfg "$CFG_FILE" \
        --longest_edge 640 \
        2>&1 | tee "$BASE_RESULT/logs/${DATASET}_${scene}_$(date +%Y%m%d_%H%M%S).log"
    

    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "ERROR: Failed to process $scene!"
        return 1
    fi
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    echo "Completed $scene in $((duration / 60))m $((duration % 60))s"
}


for scene in "${SCENES[@]}"; do
    if ! run_task "$scene"; then
        echo "Stopping pipeline due to failure on $scene"
        exit 1
    fi
done

echo "--------------------------------------------------"
echo "All tasks completed successfully at $(date)"