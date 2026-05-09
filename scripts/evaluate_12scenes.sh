#!/bin/bash

BASE_DATA="datasets/12scenes"
BASE_RESULT="outputs/12scenes"
CFG_FILE="configs/ulfloc_12scenes.yaml"

DATASET="12scenes"

SCENES=(
    "apt1/kitchen"
    "apt1/living"
    "apt2/bed"
    "apt2/kitchen"
    "apt2/living"
    "apt2/luke"
    "office1/gates362"
    "office1/gates381"
    "office1/lounge"
    "office1/manolis"
    "office2/5a"
    "office2/5b"
)

mkdir -p "$BASE_RESULT/logs"

function run_task {
    local scene=$1
    local start_time=$(date +%s)
    
    echo "--------------------------------------------------"
    echo "Starting $scene at $(date)"
    echo "Input: $BASE_DATA/$scene"
    echo "Output: $BASE_RESULT/$scene"
    

    if [ ! -d "$BASE_DATA/$scene" ]; then
        echo "ERROR: Input directory not found for $scene!"
        return 1
    fi
    
    python ulfloc.py \
        -s "$BASE_DATA/12scenes_reference_models/$scene" \
        -m "$BASE_RESULT/$scene" \
        --data_device cpu \
        --images "../../../$scene" \
        --cfg "$CFG_FILE" \
        --longest_edge 640 
    
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "ERROR: Failed to process $scene!"
        return 1
    fi
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    echo "Completed $scene in $((duration / 60))m $((duration % 60))s"
}


if [ -n "$TMUX" ]; then
    ulimit -n 65536 2>/dev/null || true
fi


for scene in "${SCENES[@]}"; do
    if ! run_task "$scene"; then
        echo "Stopping pipeline due to failure on $scene"
        exit 1
    fi
done

echo "--------------------------------------------------"
echo "All tasks completed successfully at $(date)"