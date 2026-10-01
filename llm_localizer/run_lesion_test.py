from tqdm import tqdm
import yaml
import numpy as np

import pandas as pd
import argparse
import os
from pathlib import Path
import sys
import time
# ensure project root (tomXprag) is on PYTHONPATH so "src" is importable
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import torch
from dotenv import load_dotenv

from utils import get_gpu_memory
from src.evaluate.data import TASK_TO_DATASET
import lesion_metrics

def parse_args():
    parser = argparse.ArgumentParser(description="Lesion test evaluation of LMs")
    parser.add_argument("--dataset", type=str, nargs="+", default="bigtom_backward", 
                        choices=list(TASK_TO_DATASET.keys()) + ["all", "nyu-mll/blimp"],
                        help="Name of evaluation dataset (or 'all')")
    parser.add_argument("--out_dir", type=Path, 
                        default="lesion_test_output",
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
    parser.add_argument("--use_additional_space", type=bool, default=True,
                        help="Whether to use spacing consistent with first phase of behavioral evaluation")
    parser.add_argument("--use_chat_model_formatting", type=bool, default=False,
                        help="Whether to use chat model formatting")

    # Lesion-specific arguments
    parser.add_argument("--network", type=str, default="language",
                        choices=["language", "random", "none", "theory-of-mind", "tom_mask"],
                        help="Type of network mask to apply for lesioning")
    parser.add_argument("--percentage", type=float, default=1.0,
                        help="Percentage of units to lesion")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to run on (cuda/cpu/mps)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for lesioning")
    parser.add_argument("--pooling", type=str, default="mean",
                        choices=["last-token", "mean"],
                        help="Pooling method for localization")
    parser.add_argument("--localize-range", type=str, default="100-100",
                        help="Range for localization")
    parser.add_argument("--cfg_file", type=str, default=None,
                        help="Path to YAML config file for dataset evaluation (default: src/evaluate/dataset_configs.yaml)")
    parser.add_argument("--masks_cache", type=str, default="cache_baseline",
                        help="Cache directory for lesion masks")
    parser.add_argument("--subnetwork_mask", type=str, default=None,
                        help="Saved subnetwork mask to use for theory-of-mind lesioning")
    
    args = parser.parse_args()
    
    # Resolve config file path: if not provided, use default relative to project root
    if args.cfg_file is None:
        args.cfg_file = str(project_root / "src" / "evaluate" / "dataset_configs.yaml")
    elif not os.path.isabs(args.cfg_file):
        # If relative path provided, resolve it relative to project root
        args.cfg_file = str(project_root / args.cfg_file)

    return args

def main():
    args = parse_args()
    load_dotenv()
    os.environ["HUGGINGFACE_API_TOKEN"] = os.getenv("HUGGINGFACE_API_TOKEN")
    
    # Create lesion-specific args for lesion_metrics module
    from types import SimpleNamespace
    lesion_args = SimpleNamespace(
        model_name=args.model,
        percentage=args.percentage,
        network=args.network,
        device=args.device,
        seed=args.seed,
        pooling=args.pooling,
        localize_range=args.localize_range,
        masks_cache=args.masks_cache,
        subnetwork_mask=args.subnetwork_mask
    )
    
    # Initialize lesioned model
    print(f"Setting up lesioned model {args.model}")
    model, tokenizer, device = lesion_metrics.setup_lesioned_model(lesion_args)
   
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
        
        if "nyu-mll/blimp" in dataset:
            print(f"Initializing stimuli for {dataset} dataset")
            # Make output directory.
            out_dir = Path(args.out_dir, args.masks_cache, dataset)
            os.makedirs(out_dir, exist_ok=True)

            # Evaluate model with lesioning
            print("Beginning lesion evaluation on nyu-mll/blimp")

            result = lesion_metrics.baseline_lesion_evaluation(
                ds,
                model, 
                tokenizer, 
                device,
                args.cfg_file,
                args.model,
                use_additional_space=args.use_additional_space,
                use_chat_model_formatting=args.use_chat_model_formatting,
            )
        else:      
            print(f"Initializing stimuli for {dataset} dataset")
            # Make output directory.
            if args.subnetwork_mask is not None:
                out_dir = Path(args.out_dir, args.masks_cache, args.subnetwork_mask, dataset)
            else:
                out_dir = Path(args.out_dir, args.masks_cache, dataset)
            os.makedirs(out_dir, exist_ok=True)

            # Evaluate model with lesioning
            print("Beginning lesion evaluation")

            result = lesion_metrics.run_lesion_evaluation(
                ds, 
                model, 
                tokenizer, 
                device,
                args.cfg_file,
                args.model,
                use_additional_space=args.use_additional_space,
                use_chat_model_formatting=args.use_chat_model_formatting,
            )

        # Save outputs.
        safe_model_name = args.model.split("/")[-1]
        if args.model_revision is not None:
            safe_model_name = f"{safe_model_name}_{args.model_revision}"
        if args.path_to_adapters is not None:
            safe_model_name = f"{safe_model_name}_{'-'.join(str(args.path_to_adapters).split('/')[-3:])}"
        
        # Add lesion info to filename
        lesion_info = f"network={args.network}_perc={args.percentage}_seed={args.seed}"
        out_path = Path(out_dir, f"{safe_model_name}_{lesion_info}.csv")
        result.to_csv(out_path, index=False)
        print(f"Saved model output to {out_path}!")


if __name__ == "__main__":
    time_0 = time.time()
    alloc_mem_0, res_mem_0 = get_gpu_memory()

    main()

    time_1 = time.time()
    alloc_mem_1, res_mem_1 = get_gpu_memory()

    print(f"{'='*20} Resource Usage {'='*20}")
    print(f"Time elapsed: {time_1 - time_0:.2f} seconds")
    print(f"Allocated memory: {alloc_mem_1 - alloc_mem_0} MB")
    print(f"Reserved memory: {res_mem_1 - res_mem_0} MB")
    print(f"{'='*54}")