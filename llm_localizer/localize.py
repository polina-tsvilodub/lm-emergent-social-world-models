from typing import List, Dict, Literal

import os
import torch
import argparse
import numpy as np
import transformers
import time

from tqdm import tqdm
from torch.utils.data import DataLoader
from scipy.stats import ttest_ind, false_discovery_control

from model_utils import get_layer_names, get_hidden_dim
from utils import setup_hooks, get_gpu_memory
from dataset_objects import (
    LangLocDataset, 
    TOMLocDataset, 
    MDLocDataset, 
    TomiFODataset,
    TomiSODataset,
    TOMMatchedSyntheticDataset,
    FauxpasDataset,
    PhotographSynthDataset,
    SyntheticActivationsDataset,
)

from dotenv import load_dotenv
load_dotenv()
os.getenv("HF_TOKEN")

def extract_batch(
    model: torch.nn.Module, 
    input_ids: torch.Tensor, 
    attention_mask: torch.Tensor,
    layer_names: List[str],
    pooling: str = "last-token",
):
    
    batch_activations = {layer_name: [] for layer_name in layer_names}
    hooks, layer_representations = setup_hooks(model, layer_names)
    with torch.no_grad():
        _ = model(input_ids=input_ids, attention_mask=attention_mask)

    for sample_idx in range(len(input_ids)):
        for layer_idx, layer_name in enumerate(layer_names):
            if pooling == "mean":
                activations = layer_representations[layer_name][sample_idx].mean(dim=0).cpu()
            elif pooling == "sum":
                activations = layer_representations[layer_name][sample_idx].sum(dim=0).cpu()
            else:
                activations = layer_representations[layer_name][sample_idx][-1].cpu()    
            batch_activations[layer_name] += [activations]

    for hook in hooks:
        hook.remove()

    return batch_activations

def format_chat_template(sents, tokenizer, use_chat_template, batch_size):
    print("raw sents at the beginning of format_chat_template: ", sents)
    is_chat = False
    if use_chat_template:
        if tokenizer.chat_template is not None:
            is_chat = True
            print("Using chat template for tokenizer")
            if batch_size == 1:
                if len(sents) == 2:
                    print(sents[0], sents[1])
                    sents = tokenizer.apply_chat_template(
                        [{"role": "user", "content": sents[0][0]},
                        {"role": "assistant", "content": sents[1][0]}],
                        tokenize = False,
                        continue_final_message=True,
                    )
                
                else:
                    if "Answer:" in sents[0]:
                        sents = sents[0]
                        sent = sents[0][:sents[0].rindex("Answer:")]
                        content = sents[0][sents[0].rindex("Answer:"):]
                        sents = tokenizer.apply_chat_template(
                            [{"role": "user", "content": sent},
                            {"role": "assistant", "content": content}],
                            tokenize = False,
                            continue_final_message=True,
                        )
                    else:
                        sents = tokenizer.apply_chat_template(
                            [{"role": "user", "content": sents[0]}],
                            tokenize = False,
                            continue_final_message=True,
                        )
                
            else: 
                raise ValueError("Chat template formatting currently only supported for batch_size=1")
        else:
            print("Using base model, no chat template available in tokenizer")
            if len(sents) == 2:
                sents = sents[0][0] + "\n" + sents[1][0]
    print("Sents after chat template applied: ", sents)
    return sents, is_chat

def extract_representations(
    network: str,
    pooling: str,
    model: torch.nn.Module,
    tokenizer: transformers.PreTrainedTokenizer,
    layer_names: List[str],
    hidden_dim: int,
    batch_size: int,
    device: torch.device,
    localization_dataset: str,
    use_chat_template: bool = False,
    without_answers: bool = True,
) -> Dict[str, Dict[str, np.array]]:

    single_datasets = [
        "tom_fb_synthetic", "tom_fp_synthetic", "h_synthetic", "nh_synthetic", "tom_d_synthetic", "mechanical_inference", "strategic_games_synthetic_eb", "strategic_games_eo_synthetic_paired",
        "moral_intent_tom_synthetic", "moral_intent_control_synthetic", "deceptive_mls_synthetic", "deceptive_lit_synthetic", "deceptive_iro_synthetic", "deceptive_dec_synthetic",
        "tom_d_original", "tom_fb_original", "tom_fp_original", "h_original", "nh_original", "mi_original",
        "moral_intent_original", "moral_intent_control_original",
        "strategic_games_eb_original", "strategic_games_eo_original",
        "deceptive_dec_original", "deceptive_iro_original", "deceptive_lit_original", "deceptive_mls_original"
    ]
    if network == "language":
        loc_dataset = LangLocDataset()
    elif network == "theory-of-mind":
        if (localization_dataset =="tom") or (localization_dataset =="tom_synthetic"):
            loc_dataset = TOMLocDataset(localization_dataset)
        elif (localization_dataset =="tom_matched_synthetic") or (localization_dataset =="tom_matched"):
            loc_dataset = TOMMatchedSyntheticDataset(localization_dataset)
        elif localization_dataset == "photograph_synthetic":
            loc_dataset = PhotographSynthDataset()
        elif localization_dataset == "fauxpas":
            loc_dataset = FauxpasDataset()
        elif localization_dataset == "tomi_fo":
            loc_dataset = TomiFODataset()
        elif localization_dataset == "tomi_so":
            loc_dataset = TomiSODataset()
        elif localization_dataset in single_datasets:
            loc_dataset = SyntheticActivationsDataset(localization_dataset, without_answers)
        else:
            raise ValueError(f"Unsupported localization dataset for ToM network: {localization_dataset}")
        
    elif network == "multiple-demand":
        loc_dataset = MDLocDataset()
    else:
        raise ValueError(f"Unsupported network: {network}")

    # Get the activations of the model on the dataset
    langloc_dataloader = DataLoader(loc_dataset, batch_size=batch_size, num_workers=0)

    print(f"> Using Device: {device}")

    model.eval()
    print("model's device ", model.device)
    # model.to(device)

    if network == "theory-of-mind":
        if localization_dataset in ["tom", "tom_synthetic", "tom_matched_synthetic", "tom_matched", "photograph_synthetic", "fauxpas", "tomi_fo", "tomi_so"]:
            final_layer_representations = {
                "positive": {layer_name: np.zeros((len(loc_dataset.positive), hidden_dim)) for layer_name in layer_names},
                "negative": {layer_name: np.zeros((len(loc_dataset.negative), hidden_dim)) for layer_name in layer_names}
            }
        else:
            final_layer_representations = {
                "positive": {layer_name: np.zeros((len(loc_dataset.user_context), hidden_dim)) for layer_name in layer_names}
            }
    elif (network == "language") or (network == "multiple-demand"):
        final_layer_representations = {
                "positive": {layer_name: np.zeros((len(loc_dataset.positive), hidden_dim)) for layer_name in layer_names},
                "negative": {layer_name: np.zeros((len(loc_dataset.negative), hidden_dim)) for layer_name in layer_names}
            }
    
    for batch_idx, batch_data in tqdm(enumerate(langloc_dataloader), total=len(langloc_dataloader)):
        print("batch_data ", batch_data)
        # sents = batch_data
        if (network == "theory-of-mind") and (localization_dataset in single_datasets):
            print("single dataset mode ")
            sents, is_chat = format_chat_template(batch_data, tokenizer, use_chat_template, batch_size)
        else:
            sents, is_chat = format_chat_template(batch_data[0], tokenizer, use_chat_template, batch_size)
        print("not is_chat ", not is_chat)
        sent_tokens = tokenizer(sents, padding=True, return_tensors='pt', add_special_tokens=not is_chat)

        if device != "auto":
            sent_tokens = sent_tokens.to(device)
            
        batch_real_actv = extract_batch(model, sent_tokens["input_ids"], sent_tokens["attention_mask"], layer_names, pooling)
        
        for layer_name in layer_names:
            final_layer_representations["positive"][layer_name][batch_idx*batch_size:(batch_idx+1)*batch_size] = torch.stack(batch_real_actv[layer_name]).float().numpy()
            
        if len(final_layer_representations.keys()) == 2:
            non_words = batch_data[1]
            non_words = format_chat_template(non_words, tokenizer, use_chat_template, batch_size)
            non_words_tokens = tokenizer(non_words, padding=True, return_tensors='pt')
            if device != "auto":
                non_words_tokens = non_words_tokens.to(device)
            batch_rand_actv = extract_batch(model, non_words_tokens["input_ids"], non_words_tokens["attention_mask"], layer_names, pooling)
            final_layer_representations["negative"][layer_name][batch_idx*batch_size:(batch_idx+1)*batch_size] = torch.stack(batch_rand_actv[layer_name]).float().numpy()

    return final_layer_representations

def localize(model_id: str,
    network: str,
    pooling: str,
    model: torch.nn.Module, 
    num_units: int, 
    tokenizer: transformers.PreTrainedTokenizer, 
    hidden_dim: int, 
    layer_names: List[str], 
    batch_size: int,
    seed: int,
    device: torch.device,
    percentage: float = None,
    localize_range: str = None,
    pretrained: bool = True,
    overwrite: bool = False,
    localization_dataset: str = "tom",
    use_chat_template: bool = False,
    save_raw_activations: bool = False,
    save_raw_activations_path: str = None,
    without_answers: bool = True,
):
    """
    Localize network selective units in the model.
    """

    range_start, range_end = map(int, localize_range.split("-"))

    save_path = f"{MASKS_CACHE}/{model_id}_network={network}_pooling={pooling}_range={localize_range}_perc={percentage}_nunits={num_units}_pretrained={pretrained}_dataset={localization_dataset}_chat={use_chat_template}.npy"
    save_path_pvalues = f"{MASKS_CACHE}/{model_id}_network={network}_pooling={pooling}_pretrained={pretrained}_dataset={localization_dataset}_chat={use_chat_template}_pvalues.npy"
    
    if os.path.exists(save_path) and not overwrite:
        print(f"> Loading mask from {save_path}")
        return np.load(save_path)

    representations = extract_representations(
        network=network, 
        pooling=pooling,
        model=model, 
        tokenizer=tokenizer, 
        layer_names=layer_names, 
        hidden_dim=hidden_dim, 
        batch_size=batch_size, 
        device=device,
        localization_dataset=localization_dataset,
        use_chat_template=use_chat_template,
        without_answers=without_answers,
    )
    if save_raw_activations:
        os.makedirs(os.path.dirname(save_raw_activations_path), exist_ok=True)
        
        np.save(save_raw_activations_path, representations)
        print(f"> Saved raw activations to {save_raw_activations_path}")

    if len(representations.keys()) == 1:
        print("---- ONLY ONE SET OF REPRESENTATIONS FOUND, SKIPPING LOCALIZATION -----")
        return None
    
    p_values_matrix = np.zeros((len(layer_names), hidden_dim))
    t_values_matrix = np.zeros((len(layer_names), hidden_dim))

    for layer_idx, layer_name in tqdm(enumerate(layer_names), total=len(layer_names)):

        positive_actv = np.abs(representations["positive"][layer_name])
        negative_actv = np.abs(representations["negative"][layer_name])

        t_values_matrix[layer_idx], p_values_matrix[layer_idx] = ttest_ind(positive_actv, negative_actv, axis=0, equal_var=False)
 
    def is_topk(a, k=1):
        _, rix = np.unique(-a, return_inverse=True)
        return np.where(rix < k, 1, 0).reshape(a.shape)
    
    def is_bottomk(a, k=1):
        _, rix = np.unique(a, return_inverse=True)
        return np.where(rix < k, 1, 0).reshape(a.shape)
    
    np.random.seed(seed)
    if percentage is not None:
        num_units = int((percentage/100) * hidden_dim*len(layer_names))
        print(f"> Percentage: {percentage}% --> Num Units: {num_units}")

    if localize_range is not None and range_start < range_end:
        range_start_val = np.percentile(t_values_matrix, range_start)
        range_end_val = np.percentile(t_values_matrix, range_end)
        # take random num_units from that percentile range
        mask_range = (t_values_matrix >= range_start_val) & (t_values_matrix <= range_end_val)
        total_num_units = np.prod(mask_range.shape)
        mask_range_indices = np.arange(total_num_units)[mask_range.flatten()]
        rand_indices = np.random.choice(mask_range_indices, size=num_units, replace=False)
        language_mask = np.full(total_num_units, 0)
        language_mask[rand_indices] = 1
        language_mask = language_mask.reshape(mask_range.shape)
        print(f"> Num units in range {range_start}-{range_end}: {language_mask.sum()}")
    elif localize_range and range_start == range_end and int(range_start) == 0:
        language_mask = is_bottomk(t_values_matrix, k=num_units)
    else:
        language_mask = is_topk(t_values_matrix, k=num_units)

    print(f"> Num units: {language_mask.sum()}")
    num_layers, num_units = p_values_matrix.shape
    adjusted_p_values = false_discovery_control(p_values_matrix.flatten())
    adjusted_p_values = adjusted_p_values.reshape((num_layers, num_units))

    np.save(save_path, language_mask)
    np.save(save_path_pvalues, adjusted_p_values)
    print(f"> {model_id} {network} mask cached to {save_path}")
    return language_mask

if  __name__ == "__main__":

    time_0 = time.time()
    alloc_mem_0, res_mem_0 = get_gpu_memory()

    parser = argparse.ArgumentParser(description="Localize Units in LLMs")
    parser.add_argument("--model-name", type=str, required=True, help="huggingface model name")
    parser.add_argument("--percentage", type=float, default=None, help="percentage of units to localize")
    parser.add_argument("--localize-range", type=str, default="100-100", help="percentile in which to localize, 100-100 and 0-0 indicate top and least selective units respectively")
    parser.add_argument("--network", type=str, default="language", help="network to localize")
    parser.add_argument("--localization-dataset", type=str, default="tom", help="dataset to use for localization of ToM network")
    parser.add_argument("--pooling", type=str, default="last-token", choices=["last-token", "mean"], help="token aggregation method")
    parser.add_argument("--num-units", type=int, default=None, help="number of units to localize, percentage overrides it")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--device", type=str, default=None, help="device to use")
    parser.add_argument("--untrained", action="store_true", help="use an untrained version of the model")
    parser.add_argument("--overwrite", action="store_true", help="overwrite current mask if cached")
    parser.add_argument("--use_chat_template", action="store_true", help="whether to wrap the inputs from localizer dataset into chat model formatting for respective models")
    parser.add_argument("--save_raw_activations", action="store_true", help="whether to save raw activations for further analysis")
    parser.add_argument("--masks_cache", type=str, default=None, help="directory to cache localized networks")
    parser.add_argument("--without_answers", action="store_true", help="whether to record activations before the answer option")
    
    args = parser.parse_args()

    # Get project root directory (parent of llm_localizer)
    # Path(__file__).resolve().parent.parent
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print("project_root", project_root)
    if args.masks_cache is None:
        print(args)
        print(args.masks_cache)
        raise ValueError("--masks_cache must be provided")
    MASKS_CACHE = os.path.join(project_root, args.masks_cache)
    print("MASKS_CACHE: ", MASKS_CACHE)

    os.makedirs(MASKS_CACHE, exist_ok=True)

    assert args.percentage or args.num_units, "You must either provide percentage of units to localize or number of units"
    assert args.network in {"language", "theory-of-mind", "multiple-demand"}, "Unsupported network"

    model_name = args.model_name
    pretrained = not args.untrained
    localize_range = args.localize_range
    num_units = args.num_units
    percentage = args.percentage
    pooling = args.pooling
    network = args.network
    localization_dataset = args.localization_dataset
    seed = args.seed
    batch_size = 1

    if args.device is None:
        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                device = "auto"
            else:
                device = torch.device("cuda")
        # elif torch.backends.mps.is_available():
        #     device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = args.device
    print(f">>>>> Using device: {device}")

    if ("OLMo" in model_name) or ("gemma-2" in model_name):
        dtype = torch.float32
    elif ("Llama-2" in model_name) or ("pythia" in model_name):
        dtype = torch.float16
    elif ("Llama-3" in model_name) or ("Qwen" in model_name) or ("gemma-3" in model_name) or ("Mistral" in model_name) or ("Falcon" in model_name):
        dtype = torch.bfloat16
    else:
        dtype = torch.float16

    if pretrained:
        model = transformers.AutoModelForCausalLM.from_pretrained(model_name, device_map=device, torch_dtype=dtype)
        print("Model dtype: ", model.dtype)
    else:
        model_config = transformers.AutoConfig.from_pretrained(model_name)
        model = transformers.AutoModelForCausalLM.from_config(config=model_config)

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model_name = os.path.basename(model_name)

    save_raw_activations_path = f"{args.masks_cache}/{model_name}_network={network}_pooling={pooling}_pretrained={pretrained}_dataset={localization_dataset}_chat={args.use_chat_template}_without-answers={args.without_answers}_raw_activations.npy"
    print("Created save_raw_activations_path: ", save_raw_activations_path)
    
    model_layer_names = get_layer_names(model_name)
    hidden_dim = get_hidden_dim(model_name)

    model.eval()

    localize(
        model_id=model_name,
        network=network,
        pooling=pooling,
        model=model,
        num_units=num_units,
        percentage=percentage,
        tokenizer=tokenizer,
        hidden_dim=hidden_dim,
        layer_names=model_layer_names,
        batch_size=batch_size,
        seed=seed,
        device=device,
        localize_range=localize_range,
        pretrained=pretrained,
        overwrite=args.overwrite,
        localization_dataset=localization_dataset,
        use_chat_template=args.use_chat_template,
        save_raw_activations=args.save_raw_activations,
        save_raw_activations_path=save_raw_activations_path,
        without_answers=args.without_answers,
    )

    time_1 = time.time()
    alloc_mem_1, res_mem_1 = get_gpu_memory()

    print(f"{'='*20} Resource Usage {'='*20}")
    print(f"Time elapsed: {time_1 - time_0:.2f} seconds")
    print(f"Allocated memory: {alloc_mem_1 - alloc_mem_0} MB")
    print(f"Reserved memory: {res_mem_1 - res_mem_0} MB")
    print(f"{'='*54}")