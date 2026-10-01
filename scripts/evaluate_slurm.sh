#!/bin/bash
#SBATCH ...

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

# define different model types for full evals
MODEL_TYPE=${1:-}
declare -A MODEL_GROUPS

MODEL_GROUPS["all"]="allenai/OLMo-1B-0724-hf allenai/OLMo-7B-0724-hf allenai/OLMo-7B-0724-Instruct-hf meta-llama/Llama-3.2-1B meta-llama/Llama-3.2-3B meta-llama/Llama-3.1-8B meta-llama/Llama-3.1-8B-Instruct meta-llama/Llama-3.1-70B meta-llama/Llama-3.3-70B-Instruct meta-llama/Llama-2-7b-hf meta-llama/Llama-2-7b-chat-hf meta-llama/Llama-2-13b-hf meta-llama/Llama-2-13b-chat-hf meta-llama/Llama-2-70b-hf meta-llama/Llama-2-70b-chat-hf EleutherAI/pythia-2.8b-deduped EleutherAI/pythia-1.4b-deduped EleutherAI/pythia-1b-deduped EleutherAI/pythia-6.9b-deduped EleutherAI/pythia-12b-deduped tiiuae/Falcon3-1B-Base tiiuae/Falcon3-1B-Instruct tiiuae/Falcon3-3B-Base tiiuae/Falcon3-3B-Instruct tiiuae/Falcon3-7B-Base tiiuae/Falcon3-7B-Instruct tiiuae/Falcon3-10B-Base tiiuae/Falcon3-10B-Instruct Qwen/Qwen2.5-0.5B Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-1.5B Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen2.5-3B Qwen/Qwen2.5-3B-Instruct Qwen/Qwen2.5-7B Qwen/Qwen2.5-7B-Instruct Qwen/Qwen2.5-14B Qwen/Qwen2.5-14B-Instruct Qwen/Qwen2.5-32B Qwen/Qwen2.5-32B-Instruct Qwen/Qwen2.5-72B Qwen/Qwen2.5-72B-Instruct mistralai/Mistral-7B-Instruct-v0.3" 
MODEL_GROUPS["revision"]="meta-llama/Llama-3.1-8B-Instruct tiiuae/Falcon3-7B-Base tiiuae/Falcon3-7B-Instruct Qwen/Qwen2.5-7B Qwen/Qwen2.5-7B-Instruct"
# MODEL_GROUPS["all"]="meta-llama/Llama-3.2-1B meta-llama/Llama-3.2-3B meta-llama/Llama-3.1-8B meta-llama/Llama-3.1-8B-Instruct meta-llama/Llama-3.1-70B meta-llama/Llama-3.3-70B-Instruct tiiuae/Falcon3-1B-Base tiiuae/Falcon3-1B-Instruct tiiuae/Falcon3-3B-Base tiiuae/Falcon3-3B-Instruct tiiuae/Falcon3-7B-Base tiiuae/Falcon3-7B-Instruct tiiuae/Falcon3-10B-Base tiiuae/Falcon3-10B-Instruct Qwen/Qwen2.5-0.5B Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-1.5B Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen2.5-3B Qwen/Qwen2.5-3B-Instruct Qwen/Qwen2.5-7B Qwen/Qwen2.5-7B-Instruct Qwen/Qwen2.5-14B Qwen/Qwen2.5-14B-Instruct Qwen/Qwen2.5-32B Qwen/Qwen2.5-32B-Instruct Qwen/Qwen2.5-72B Qwen/Qwen2.5-72B-Instruct mistralai/Mistral-7B-Instruct-v0.3" 

MODEL_GROUPS["Pythia"]="EleutherAI/pythia-6.9b-deduped EleutherAI/pythia-2.8b-deduped EleutherAI/pythia-1.4b-deduped EleutherAI/pythia-1b-deduped"

MODEL_GROUPS["Llama-v2"]="meta-llama/Llama-2-7b-hf"
MODEL_GROUPS["Llama-v3"]="meta-llama/Llama-3.1-8B meta-llama/Llama-3.2-1B meta-llama/Llama-3.2-3B"

MODEL_GROUPS["Qwen2.5"]="Qwen/Qwen2.5-0.5B Qwen/Qwen2.5-1.5B Qwen/Qwen2.5-3B Qwen/Qwen2.5-7B"
MODEL_GROUPS["Qwen1.5"]="Qwen/Qwen1.5-0.5B Qwen/Qwen1.5-1.8B Qwen/Qwen1.5-4B Qwen/Qwen1.5-7B"

MODEL_GROUPS["OLMo-2"]="allenai/OLMo-2-0425-1B allenai/OLMo-2-0425-1B-Instruct allenai/OLMo-2-1124-7B allenai/OLMo-2-1124-7B-Instruct allenai/OLMo-2-1124-13B allenai/OLMo-2-1124-13B-Instruct"
MODEL_GROUPS["OLMo-1"]="allenai/OLMo-1B-0724-hf allenai/OLMo-7B-0724-hf"

MODEL_GROUPS["Gemma-3"]="google/gemma-3-270m google/gemma-3-1b-pt google/gemma-3-4b-pt"
MODEL_GROUPS["Gemma-2"]="google/gemma-2-27b"

MODEL_GROUPS["Mistral"]="mistralai/Mistral-7B-Instruct-v0.3"

MODEL_GROUPS["Falcon3"]="tiiuae/Falcon3-1B-Base tiiuae/Falcon3-3B-Base tiiuae/Falcon3-7B-Base"

MODEL_GROUPS["large"]="EleutherAI/pythia-12b-deduped meta-llama/Llama-2-13b-hf Qwen/Qwen2.5-14B google/gemma-3-12b-pt tiiuae/Falcon3-10B-Base"
MODEL_GROUPS["chat"]="meta-llama/Llama-2-7b-chat-hf meta-llama/Llama-2-13b-chat-hf"
MODEL_GROUPS["instruct_1"]="allenai/OLMo-7B-0724-Instruct-hf meta-llama/Llama-3.1-8B-Instruct Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen2.5-3B-Instruct Qwen/Qwen2.5-7B-Instruct"
MODEL_GROUPS["instruct_2"]="tiiuae/Falcon3-1B-Instruct tiiuae/Falcon3-3B-Instruct tiiuae/Falcon3-7B-Instruct tiiuae/Falcon3-10B-Instruct"
MODEL_GROUPS["instruct_3"]="google/gemma-3-1b-it google/gemma-3-4b-it google/gemma-3-12b-it"
MODEL_GROUPS["debug"]="meta-llama/Llama-2-13b-chat-hf meta-llama/Llama-3.1-8B-Instruct mistralai/Mistral-7B-Instruct-v0.3"
MODEL_GROUPS["selected"]="meta-llama/Llama-3.1-8B tiiuae/Falcon3-7B-Base tiiuae/Falcon3-7B-Instruct Qwen/Qwen2.5-7B Qwen/Qwen2.5-7B-Instruct tiiuae/Falcon3-10B-Base tiiuae/Falcon3-10B-Instruct google/gemma-2-27b-it google/gemma-2-27b Qwen/Qwen2.5-32B Qwen/Qwen2.5-32B-Instruct meta-llama/Llama-3.1-70B meta-llama/Llama-3.3-70B-Instruct Qwen/Qwen2.5-72B Qwen/Qwen2.5-72B-Instruct"
MODEL_GROUPS["remaining"]="EleutherAI/pythia-1.4b-deduped EleutherAI/pythia-12b-deduped EleutherAI/pythia-1b-deduped EleutherAI/pythia-2.8b-deduped EleutherAI/pythia-6.9b-deduped Qwen/Qwen2.5-0.5B Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-1.5B Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen2.5-14B Qwen/Qwen2.5-14B-Instruct Qwen/Qwen2.5-3B Qwen/Qwen2.5-3B-Instruct allenai/OLMo-1B-0724-hf allenai/OLMo-7B-0724-Instruct-hf allenai/OLMo-7B-0724-hf meta-llama/Llama-2-70b-chat-hf meta-llama/Llama-2-70b-hf meta-llama/Llama-2-7b-chat-hf meta-llama/Llama-2-7b-hf meta-llama/Llama-3.2-1B tiiuae/Falcon3-1B-Base tiiuae/Falcon3-1B-Instruct tiiuae/Falcon3-3B-Base tiiuae/Falcon3-3B-Instruct"
MODEL_GROUPS["revisions_m"]="meta-llama/Llama-3.1-8B meta-llama/Llama-2-13b-hf meta-llama/Llama-2-13b-chat-hf tiiuae/Falcon3-10B-Base tiiuae/Falcon3-10B-Instruct Qwen/Qwen2.5-32B Qwen/Qwen2.5-32B-Instruct"
MODEL_GROUPS["revisions_l"]="Qwen/Qwen2.5-72B Qwen/Qwen2.5-72B-Instruct meta-llama/Llama-3.1-70B meta-llama/Llama-3.3-70B-Instruct google/gemma-2-9b google/gemma-2-9b-it google/gemma-2-27b google/gemma-2-27b-it"
# --- Resolve models ---
if [[ -z "${MODEL_TYPE}" ]]; then
  echo "Usage: $0 <ModelType|ModelName>"
  echo "Available types: ${!MODEL_GROUPS[@]}"
  exit 1
fi

if [[ -n "${MODEL_GROUPS[$MODEL_TYPE]:-}" ]]; then
  MODELS=${MODEL_GROUPS[$MODEL_TYPE]}
else
  # assume raw model name
  MODELS=$MODEL_TYPE
fi
# meta-llama/Llama-2-7b-hf meta-llama/Llama-2-7b-chat-hf EleutherAI/pythia-12b-deduped 
# hufloyd_deceits hufloyd_indirectspeech hufloyd_irony hufloyd_maxims hufloyd_metaphor ludwig triangle-copa 
# tinytom-v3_forward tinytom-v3_backward tinytom-v4_forward tinytom-v4_backward tomi_tb tomi_fb tomi_sofb opentom_location_cg_fo opentom_location_cg_so opentom_location_cg_fo_long opentom_location_cg_so_long opentom_location_fg_fo_new opentom_location_fg_so_new opentom_multihop_fo opentom_multihop_so opentom_attitude
for MODEL in $MODELS; do
    for phen in story_analogies verbal_analogies entity_tracking snli; do
        echo ">>> Running "$MODEL" on $phen"
        bash scripts/evaluate_model.sh "$MODEL" $phen
    done
done
