#!/bin/bash
#SBATCH ...
#SBATCH --array=0-4
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

NETWORKS=("theory-of-mind")

SUBNETWORK_MASKS=(
"tom|theory-of-mind"
"tom|random"
"tom|theory-of-mind-conjunctive"
"tom|random-conjunctive"
"moral_intent|theory-of-mind"
"moral_intent|random"
"strategic_games|theory-of-mind"
"strategic_games|random"
"deceptive_communication|theory-of-mind"
"deceptive_communication|random"
"deceptive_communication|theory-of-mind-conjunctive"
"deceptive_communication|random-conjunctive"
"all|theory-of-mind"
"all|random"
"all|theory-of-mind-conjunctive"
"all|random-conjunctive"
)
MASKS_CACHE="cache_tom_fb_synthetic" 

MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

echo ">>> Model: $MODEL"
echo ">>> Datasets: ${DATASETS[@]}"

for NETWORK in "${NETWORKS[@]}"; do
  for SUBNETWORK in "${SUBNETWORK_MASKS[@]}"; do
    # Extract datasets for this model from JSON file using jq
    JSON_FILE="llm_localizer/scripts/models_selected_datasets_covered_above_chance.json"
    if [[ ! -f "$JSON_FILE" ]]; then
      echo "ERROR: Missing JSON file $JSON_FILE" >&2
      exit 1
    fi
    mapfile -t DATASETS < <(jq -r --arg model "$MODEL" '.[$model].datasets[]?' "$JSON_FILE")
    # alternatively, we tested additional general-reasoning datasets (next line); if used, comment out the seven lines above
    # DATASETS=("snli" "story_analogies" "verbal_analogies" "entity_tracking")
    for DATASET in "${DATASETS[@]}"; do
      echo ">>> Running "$MODEL" ablation and evals on "$DATASET" dataset with network "$NETWORK" and subnetwork "$SUBNETWORK""
      python llm_localizer/run_lesion_test.py \
        --dataset "$DATASET" \
        --model "$MODEL" \
        --network "$NETWORK" \
        --percentage 1 \
        --pooling last-token \
        --use_additional_space True \
        --use_chat_model_formatting True \
        --out_dir lesion_test_output \
        --masks_cache "$MASKS_CACHE" \
        --subnetwork_mask "$SUBNETWORK"
    done
  done
done
