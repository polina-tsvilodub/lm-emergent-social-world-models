import pandas as pd
from pathlib import Path
from datasets import load_dataset, get_dataset_config_names


class DatasetBase():
    """Base class for evaluation datasets."""
    def __init__(self, name: str, **kwargs):
        self.name = name
        self.items = self.load_items(**kwargs)

    def load_items(self, _):
        raise NotImplementedError

class CSVDataset(DatasetBase):
    """Class for datasets based on a CSV file in a local folder."""
    def load_items(self, stimuli_folder: str = "behavioral_eval_stimuli/log_p_curated"):
        # for previously wrangled items available in the repository, load from local CSV files
        try:
            df = pd.read_csv(
                Path(stimuli_folder, f"{self.name}.csv")
            )
        # otherwise, pull from huggingface
        except FileNotFoundError:
            try:
                if "imppres" in self.name:
                    # collecting all subsets and splits
                    cfgs = get_dataset_config_names(self.name)
                    ds_subsets = [
                        load_dataset(self.name, config)
                        for config in cfgs
                    ]
                    dfs = [
                        ds[split].to_pandas()
                        for ds in ds_subsets
                        for split in ds.keys()
                    ]
                    df = pd.concat(dfs)
                else:
                    try:
                        df = load_dataset(self.name, split="train")
                    except Exception as e:
                        print("Using trust_remote_code=True to load dataset", e)
                        try:
                            df = load_dataset(self.name, split="validation", trust_remote_code=True)
                        except Exception as e:
                            print("Using split='train' to load dataset", e)
                            df = load_dataset(self.name, split="train", trust_remote_code=True)
                    df = df.to_pandas()
            except Exception as e:
                raise RuntimeError(
                    f"Could not load dataset {self.name} on Hugging Face: {e}"
                )
            
        self.n_items = len(df)
        return df

# list of tasks, mapping task names to HF dataset IDs if the dataset is from HF, otherwise to empty string 
TASKS = [
    "epitome_rm",
    "epitome_fb",
    "epitome_si",
    "ewok_agent-properties",
    "ewok_social-interactions",
    "ewok_social-properties",
    "simpletom_behavior",
    "bigtom_forward",
    "bigtom_backward",
    "tinytom-v3_forward",
    "tinytom-v3_backward",
    "tinytom-v4_forward",
    "tinytom-v4_backward",
    "hufloyd_deceits",
    "hufloyd_indirectspeech",
    "hufloyd_irony",
    "hufloyd_maxims",
    "hufloyd_metaphor",
    "hufloyd_coherence",
    "hufloyd_humour",
    "ludwig",
    "triangle-copa",
    "social_iqa",
    "fauxpas_eai_q1_and_q4",
    "tomi_tb",
    "tomi_fb",
    "tomi_sofb",
    "opentom_location_cg_fo",
    "opentom_location_cg_so",
    "opentom_location_cg_fo_long",
    "opentom_location_cg_so_long",
    "opentom_location_fg_fo_new",
    "opentom_location_fg_so_new",
    "opentom_multihop_fo",
    "opentom_multihop_so",
    "opentom_attitude",
    "pub_deictic_qa",
    "pub_sarcasm",
    "pub_agreement",
    "pub_indirectness_interpretation",
    "pub_indirectness_classification",
    "imppres_implicature",
    "imppres_presupposition",
    "sarcv2",
    "emobench_understanding_emotion",
    "emobench_understanding_cause",
    "emobench_application",
    "tomi_sapEtAl_first_order",
    "tom_localizer",
    "blimp",
    "snli",
    "story_analogies",
    "verbal_analogies",
    "entity_tracking"
]

TASK_TO_DATASET = {task: CSVDataset for task in TASKS}
