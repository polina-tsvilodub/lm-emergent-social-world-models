# On Emergent Social World Models — Evidence for Functional Integration of Theory of Mind and Pragmatic Reasoning in Language Models

This repository contains all materials for the paper by Tsvilodub et al. (2026) [On Emergent Social World Models — Evidence for Functional Integration of Theory of Mind and Pragmatic Reasoning in Language Models](https://arxiv.org/abs/2602.10298).

## Installation 

To run the code, clone the repository, navigate into the cloned directory via `cd lm-emergent-social-world-models`, and install the requirements e.g. via conda as:

```{python}
conda create -m olmo_env -f requirements.txt
```

## Running the code

The code can be run by directly executing the respective python scripts as described next, or on an HPC cluster. Respective scripts for a slurm cluster are documented below in the detailed docs. 

The main functions of the code and respective entry points are:

### 1. Running evaluations of existing models on preprocessed datasets:

```{python}
python src/evaluate/run_behavioral_evaluation.py --model <model of your choice> --dataset <name of your ds or 'all'> 
```
where the model name is the Huggingface ID of the model and the dataset name is the name of an available dataset as listed in `TASKS` in `src/evaluate/data.py`.

### 2. Running functional subnetwork localizations:

The procedure consists of several steps: (1) constructing a localizer suite, (2) retrieving activations of a model's units, (3) extracting binary masks indicating which neurons pass the target criterion (e.g., most or least active neurons under a particular statistical analysis).

The following command will primarily produce a .npy file with all activations in a location `save_raw_activations_path` specified in `llm_localizer/localize.py`, and dump binary masks and p-values based on the procedure from the [original code](https://github.com/BKHMSI/llm-localization) (deprecated).

To (1) construct localizers through synthetic generation, refer to `src/synthetic_localizer_generation`.

To (2) save the activations of the model, run:

```{python}
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
```

where 

- `model-name` is a huggingface ID of the model 
- `percentage` is the percentage of neurons meeting the selection criterion to include in the identified subnetwork
- `network` can be one of `["language", "theory-of-mind"]`
- `localize-range` determines the percentile range of units to localize.
where LOCALIZATION_DATASET can be one of
- `pooling` determines how the signal across tokens passed as input in the localizer item is aggregated (`["last-token", "mean"]`)
- `localization-dataset` is the name of the localizer suite (available choices are documentd in detail in `llm_localizer/stimuli/readme.md`)
- `use_chat_template` (bool) indicating whether the chat template should be used for models that have one
- `overwrite` (bool) indicates whether to overwrite current mask if one is already cached
- `save_raw_activations` (bool) indicates that all activations (each unit, each item) should be saved
- `masks_cache` provides the location where the masks and p-value files according to the original procedure are saved
- `without_answers` (bool) indicates that the actual answers to the localization tasks should NOT be appended to the trigger.

To (3) generate masks that can be applied for ablation according to simple and conjunctive statistical analyses, refer to `llm_localizer/netowrk_analytics.ipynb`.  


### 3. Ablating localized subnetworks and running evaluations of the ablated models:

Assuming available .npy masks in the directory `MASKS_CACHE`, the following will apply the mask (thereby ablating the subnetwork) and evaluate the ablated model on `DATASET`:

```{python}
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
```
where the arguments are similar as above in the localization step, with the new arguments

- `use_additional_space` which appends a space to the end of the context *and* the scored option 
- `out_dir` specifies the output directory were the evaluation results are saved
- `subnetwork_mask` specifies the type of the subnetwork to be ablated. Choices are: `["tom|theory-of-mind", "tom|random", "tom|theory-of-mind-conjunctive", "tom|random-conjunctive", "moral_intent|theory-of-mind", "moral_intent|random", "strategic_games|theory-of-mind", "strategic_games|random", "deceptive_communication|theory-of-mind", "deceptive_communication|random", "deceptive_communication|theory-of-mind-conjunctive", "deceptive_communication|random-conjunctive", "all|theory-of-mind", "all|random", "all|theory-of-mind-conjunctive", "all|random-conjunctive"]`.

See the next sections for details on how to add new evaluation benchmarks and localizers.


## Repository structure

The localization and ablation code is adapted from [this](https://github.com/BKHMSI/llm-localization) repo.

The detailed code documentation is:

- `llm_localizer/`: top-level directory containing all code related to the functional subnetowrks.
  - `scripts/`: directory with slurm scripts for scheduling localization and subnetowrk ablation jobs
  - `stimuli/`: directory with localizer suites
  - `dataset_ibjects.py`: implementation of dataset objects for localizer suites
  - `lesion_metrics.py`: key script applying an ablation to a pretrained model and evaluating it on a given dataset through conditional log probability scoring
  - `localize.py`: key script for retrieving activations of a model on a given localizer suite
  - `model_utils.py`: utils
  - `network_analytics.ipynb`: notebook for creating masks (calculating subnetworks) based on activations with different statistical procedures. Produces both target and control subnetwork masks.
  - `run_lesion_test.py`: entrypoint for evaluating ablated models
  - `utils.py`: utils
- `scripts`: directory with slurm scripts for evaluations jobs (intact models) 
- `src`: directory with scripts for working with intact models and preparing data
  - `evaluate/`: scripts for model evaluation
  - `preprocess/`: notebook for preparing evaluation datasets
  - `synthetic_localizer_generation/`: scripts for generating new localizer materials with GPT-5 based on few-shot examples
- `behvaioral_eval_stimuli/`: directory with datasets in csv files on which the models were evaluates
- `behavioral_eval_outputs/`: directory with predictions of intact models on each dataset (each has a directory)
- `lesion_test_output/`: directory with predictions of ablated models on selected dataset (each has a directory). Each directory therein contains results of the ablation of the respective subnetwork. In each directroy, results are organized by dataset.
- `analysis/`:
  - `analyse_ablations.ipynb`: notebook for wrangling and plotting results of the subnetwork ablations
  - `analyse_correlations.ipynb`: notebook for wrangling and plotting results of the intact model evaluations and correlating ToM and pragmatic performance of LLMs
  - `baseline_stats.Rmd`: all regression modeling reported in the paper
  - `lookup_subtlex_unigrams.py`: script for evaluating the frequencies of the words in the localizer suites for quality control of the synthetic localizers
  - `plots.ipynb`: notebook for plotting the distributions of the subnetworks across model layers

## Extenting the evaluation datasets and localizers

### Adding new evaluation datasets

All evaluations were performed on multiple-choice datasets, via scoring the conditional probability of each answer option, given the context. The evaluated datasets (all in `behavioral_eval_stimuli/log_p_curated/`) were preprocessed into wide csv files, with separate columns for each answer option. The specification of the format of each dataset for scoring is handled through the config file `src/evaluate/dataset_configs.yaml`. To add a dataset, preprocess it (e.g. in `src/preprocess/`) and complete the following steps:

- place a csv into `behavioral_eval_stimuli/log_p_curated/`. The minimal format for the csv are columns (1) containing the context to be conditioned on (e.g., "context", but can be several columns), (2) containing the correct answer options (e.g., "correct") and (3) containing the incorrect answer option (e.g., "incorrect").
- add a new item to the dict of tasks `TASKS` in `src/evaluate/data.py`
- add the dataset specs to dataset_configs.yaml (format should be self-explanatory; if needed, new keys can be added, but then the metrics in the next step likely also need to be adjusted)
- if needed, adjust `run_scorer()` in `src/evaluate/metrics.py` to format the context and options correctly.


### Adding new localizers

TODO.

## Citation

```
@inproceedings{tsvilodub-etal-2026-emergent,
    title = "On Emergent Social World Models {---} Evidence for Functional Integration of Theory of Mind and Pragmatic Reasoning in Language Models",
    author = "Tsvilodub, Polina  and
      Klumpp, Jan-Felix  and
      Mohammadpour, Amir  and
      Hu, Jennifer  and
      Franke, Michael",
    editor = "Liakata, Maria  and
      Moreira, Viviane P.  and
      Zhang, Jiajun  and
      Jurgens, David",
    booktitle = "Proceedings of the 64th Annual Meeting of the {A}ssociation for {C}omputational {L}inguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2026",
    address = "San Diego, California, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.acl-long.1735/",
    doi = "10.18653/v1/2026.acl-long.1735",
    pages = "37382--37420",
    ISBN = "979-8-89176-390-6",
    abstract = "This paper investigates whether LMs recruit shared computational mechanisms for general Theory of Mind (ToM) and language-specific pragmatic reasoning in order to contribute to the general question of whether LMs may be said to have emergent ``social world models'', i.e., representations of mental states that are repurposed across tasks (the functional integration hypothesis). Using behavioral evaluations and causal-mechanistic experiments via functional localization methods inspired by cognitive neuroscience, we analyze LMs' performance across seven subcategories of ToM abilities (Beaudoin et al., 2020) on a substantially larger localizer dataset than used in prior like-minded work. Results from stringent hypothesis-driven statistical testing offer suggestive evidence for the functional integration hypothesis, indicating that LMs may develop interconnected ``social world models'' rather than isolated competencies. This work contributes novel ToM localizer data, methodological refinements to functional localization techniques, and empirical insights into the emergence of social cognition in artificial systems."
}
```