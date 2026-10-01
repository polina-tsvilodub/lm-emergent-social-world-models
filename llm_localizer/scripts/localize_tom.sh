#!/bin/bash
#SBATCH ...
#SBATCH --array=0-9 

echo 'Running evaluation script'


# print the current working directory
# (should be the directory of the script)
echo "Current working directory:"
echo $(pwd)

# Load conda
module load devel/miniforge

conda deactivate
# Activate the conda environment
conda activate olmo_env

export HF_HOME=""
export HF_TOKEN=""

echo "Conda environment activated:"
echo $(conda env list)
echo " "
echo "Python version:"
echo $(which python)
echo " "

MODELS=(
# large models
"meta-llama/Llama-3.1-70B"              #0
"meta-llama/Llama-3.3-70B-Instruct"     #1

"Qwen/Qwen2.5-72B"                      #2
"Qwen/Qwen2.5-72B-Instruct"             #3
# medium models
"google/gemma-2-27b-it"                 #4
"google/gemma-2-27b"                    #5

"google/gemma-2-9b-it"                  #6
"google/gemma-2-9b"                     #7

"tiiuae/Falcon3-10B-Base"               #8  
"tiiuae/Falcon3-10B-Instruct"           #9

"Qwen/Qwen2.5-32B"                      #10
"Qwen/Qwen2.5-32B-Instruct"             #11

# small models
"meta-llama/Llama-3.1-8B"               #12
"meta-llama/Llama-3.1-8B-Instruct"      #13

"tiiuae/Falcon3-7B-Base"                #14
"tiiuae/Falcon3-7B-Instruct"            #15
"tiiuae/Falcon3-10B-Base"               #16
"tiiuae/Falcon3-10B-Instruct"           #17

"Qwen/Qwen2.5-7B"                       #18
"Qwen/Qwen2.5-7B-Instruct"              #19
)

LOCALIZATION_DATASETS=(
# localizer conditions for the LatentBeliefs localization suite
"tom_fb_synthetic" 
"tom_fp_synthetic" 
"h_synthetic" 
"nh_synthetic" 
"tom_d_synthetic" 
"mechanical_inference" 
# localizer conditions for the GameBeliefs localization suite
"strategic_games_synthetic_eb" 
"strategic_games_eo_synthetic_paired"
# localizer conditions for the MoralIntent localization suite
"moral_intent_tom_synthetic"
"moral_intent_control_synthetic"
# localizer conditions for the CommunicativeIntent localization suite
"deceptive_mls_synthetic"
"deceptive_lit_synthetic"
"deceptive_iro_synthetic"
"deceptive_dec_synthetic"
)

NETWORKS=("theory-of-mind")

MASKS_CACHE="cache"

MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

for LOCALIZATION_DATASET in "${LOCALIZATION_DATASETS[@]}"; do
    for NETWORK in "${NETWORKS[@]}"; do
        case "$NETWORK" in
            "language")
            echo ">>> Running "$MODEL" localization on "$LOCALIZATION_DATASET" dataset"
            python llm_localizer/localize.py \
                --model-name "$MODEL" \
                --percentage 1 \
                --network "$NETWORK" \
                --localize-range 100-100 \
                --pooling last-token \
                --localization-dataset "$LOCALIZATION_DATASET" \
                --use_chat_template \
                --overwrite \
                --save_raw_activations \
                --masks_cache "$MASKS_CACHE"
        ;;
            *)
            echo ">>> Running "$MODEL" localization on "$LOCALIZATION_DATASET" dataset"
            python llm_localizer/localize.py \
                --model-name "$MODEL" \
                --percentage 1 \
                --network "$NETWORK" \
                --localize-range 100-100 \
                --pooling last-token \
                --localization-dataset "$LOCALIZATION_DATASET" \
                --use_chat_template \
                --overwrite \
                --save_raw_activations \
                --masks_cache "$MASKS_CACHE" \
                --without_answers
            ;;
        esac
    done
done

