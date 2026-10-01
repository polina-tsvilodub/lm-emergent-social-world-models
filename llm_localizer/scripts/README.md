* `localize_tom.sh`: start localization job (can be parallelized over models via starting and array job). Its default result are .npy files recording all model unit activations for all stimuli. The following parameters in the script are important:
  * `HF_HOME`, `HF_TOKEN`: set them appropriately
  * `MODELS` defines the set of evaluated models 
  * `LOCALIZATION_DATASETS` defines the set of localization datasets (one per localization conditions)
  * `NETWORKS`: defines the network to be localized. Only `theory-of-mind` is used in the paper.
  * `MASKS_CACHE`: location where the result .npy files are dumped.
  * the following arguments are passed:
    * `model-name` is a huggingface ID of the model 
    * `percentage` is the percentage of neurons meeting the selection criterion to include in the identified subnetwork
    * `network` can be one of `["language", "theory-of-mind"]`
    * `localize-range` determines the percentile range of units to localize.
    * `pooling` determines how the signal across tokens passed as input in the localizer item is aggregated (`["last-token", "mean"]`)
    * `localization-dataset` is the name of the localizer suite (available choices are documentd in detail in `llm_localizer/stimuli/readme.md`)
    * `use_chat_template` (bool) indicating whether the chat template should be used for models that have one
    * `overwrite` (bool) indicates whether to overwrite current mask if one is already cached
    * `save_raw_activations` (bool) indicates that all activations (each unit, each item) should be saved
    * `masks_cache` provides the location where the masks and p-value files according to the original procedure are saved
    * `without_answers` (bool) indicates that the actual answers to the localization tasks should NOT be appended to the trigger.

* `models_selected_datasets_covered_above_chance.json`: dict containing, for each evaluated model, a selection of datasets for evaluation on which the intact model performed robustly above chance. Used evals of ablated models in the next script.
* `run_lesion_test.sh` start job testing the ablated models(can be parallelized via starting and array job). NOTE: The latter requiretha localization was already performed and subnetwork masks are available. The following parameters in the script are important:
  * `HF_HOME`, `HF_TOKEN`: set them appropriately
  * `MODELS` defines the set of evaluated models. Must match the set of models for which localizations were performed. 
  * `SUBNETWORK_MASKS` defines the set of localizations, which must be generated through `network_analytics.ipynb`. There are 16 possible localizations (8 critical, 8 least active controls).
  * `NETWORKS`: defines the network to be localized. Only `theory-of-mind` is used in the paper.
  * `MASKS_CACHE`: location where the masks created in `network_analytics.ipynb` can be found.
  * the following script-specific arguments are passed (the rest match the script above):
    * `use_additional_space` which appends a space to the end of the context *and* the scored option
    * `out_dir` specifies the output directory were the evaluation results are saved
    * `subnetwork_mask` specifies the type of the subnetwork to be ablated. Choices are: `["tom|theory-of-mind", "tom|random", "tom|theory-of-mind-conjunctive", "tom|random-conjunctive", "moral_intent|theory-of-mind", "moral_intent|random", "strategic_games|theory-of-mind", "strategic_games|random", "deceptive_communication|theory-of-mind", "deceptive_communication|random", "deceptive_communication|theory-of-mind-conjunctive", "deceptive_communication|random-conjunctive", "all|theory-of-mind", "all|random", "all|theory-of-mind-conjunctive", "all|random-conjunctive"]`.