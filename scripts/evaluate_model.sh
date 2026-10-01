#!bin/bash

# Change this path to point to wherever your Huggingface models are stored
MODEL=$1
DATASET=$2

python src/evaluate/run_behavioral_evaluation.py \
    --model $MODEL \
    --dataset $DATASET \
    --out_dir behavioral_eval_output/original
