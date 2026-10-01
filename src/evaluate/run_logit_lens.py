from tqdm import tqdm
import yaml
import numpy as np
import time

import pandas as pd
import argparse
import os
from pathlib import Path
import torch
from dotenv import load_dotenv

from model import nnsightLM
from data import TASK_TO_DATASET
import metrics
from utils import get_gpu_memory

def parse_args():
    parser = argparse.ArgumentParser(description="Behavioral evaluation of LMs")
    parser.add_argument("--dataset", type=str, nargs="+", default="bigtom_backward", 
                        choices=list(TASK_TO_DATASET.keys()) + ["all"],
                        help="Name of evaluation dataset (or 'all')")
    parser.add_argument("--out_dir", type=Path, 
                        default="logit_lens_output",
                        help="Path to directory where output files will be saved")
    parser.add_argument("--model", type=str, default="gpt2",
                        help="Huggingface model identifier")
    parser.add_argument("--tokenizer", type=str, default=None,
                        help="Huggingface model identifier for tokenizer")
    parser.add_argument("--cache_dir", type=Path, default=None,
                        help="Path to Huggingface cache directory")
    parser.add_argument("--path_to_adapters", type=str, default=None,
                        help="Adapter files for testing LORA models at intermediate training stages")
    parser.add_argument("--model_revision", type=str, default=None,
                        help="Revision of the model to use for intermediate checkpoint evals")
    args = parser.parse_args()
    return args

def main():

    # Get GPU memory and time at start of evaluation
    mem_alloc_start, mem_reserved_start = get_gpu_memory()
    start_time = time.time()

    args = parse_args()
    load_dotenv()
    os.environ["HUGGINGFACE_API_TOKEN"] = os.getenv("HUGGINGFACE_API_TOKEN")
    
    # Initialize model.
    print(f"Initializing Huggingface model {args.model}")
    
    # initialize model for logit lens
    m = nnsightLM(
        args.model
    )
   
    # get stimuli
    stimuli_folder = "./behavioral_eval_stimuli/log_p_curated"

    # Evaluate model on each specified dataset.
    if args.dataset == ["all"]:
        print("Evaluating model on *all* datasets")
        datasets = TASK_TO_DATASET.keys()
    else:
        datasets = args.dataset

    for dataset in datasets:
        try:
            ds = TASK_TO_DATASET[dataset](dataset, stimuli_folder=stimuli_folder)
        except KeyError:
            raise ValueError(f"Dataset {dataset} is not supported or not implemented yet; skipping now.")
        
        print(f"Initializing stimuli for {dataset} dataset")
        # Make output directory.
        out_dir = Path(args.out_dir, dataset)
        os.makedirs(out_dir, exist_ok=True)

        # Evaluate model.
        print("Beginning logit lens evaluation")

        result = metrics.run_logit_lens_evaluation(ds, m)

        # Save outputs.
        # TODO: make sure that the result is formatted such that the saved results can be processed
        
        safe_model_name = args.model.split("/")[-1]
        if args.model_revision is not None:
            safe_model_name = f"{safe_model_name}_{args.model_revision}"
        if args.path_to_adapters is not None:
            safe_model_name = f"{safe_model_name}_{'-'.join(str(args.path_to_adapters).split('/')[-3:])}"
        out_path = Path(out_dir, f"{safe_model_name}.csv")
        result.to_csv(out_path, index=False)
        print(f"Saved model output to {out_path}!")

        # Get GPU memory and time at the end of evaluation
        mem_alloc_end, mem_reserved_end = get_gpu_memory()
        end_time = time.time()

        print(f"\n==== A Summary of the logitlens run ====")
        print(f"\nGPU Memory Usage:")
        print(f"Allocated: {mem_alloc_end-mem_alloc_start:.2f} GB")
        print(f"Reserved: {mem_reserved_end-mem_reserved_start:.2f} GB")
        print(f"\nTime taken: {end_time - start_time:.2f} seconds")
        print(f"{'='*40}")


if __name__ == "__main__":
    main()
