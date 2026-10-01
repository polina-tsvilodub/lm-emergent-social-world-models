import os
import sys
from pathlib import Path
import pandas as pd
import ast
import yaml
from tqdm import tqdm
import torch
import argparse
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv
from datasets import load_dataset, get_dataset_config_names

load_dotenv()
os.getenv("HF_TOKEN")

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.evaluate.utils import special_chat_formatting

# Model-agnostic layer wrapper to apply masks --- NOTE: Test this promptly before use!

class MaskedLayer(torch.nn.Module):
    def __init__(self, layer: torch.nn.Module, mask_row: torch.Tensor):
        super().__init__()
        self.layer = layer
        self.register_buffer("mask", mask_row.view(1, 1, -1))

    def forward(self, *args, **kwargs):
        out = self.layer(*args, **kwargs)
        if isinstance(out, tuple):
            h = out[0] * self.mask
            # NOTE: renorm applied after mask application to all layers in the original code: hidden_states = self.norm(hidden_states) 
            # -> check if norm is applied here in the same way
            return (h,) + out[1:] 

        if hasattr(out, "last_hidden_state"):
            out.last_hidden_state = out.last_hidden_state * self.mask
            return out

        return out * self.mask


def find_decoder_blocks(model: torch.nn.Module):
    # Llama / Mistral / Gemma / Qwen / Phi / OLMo
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers

    # GPT-2 / Falcon
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h

    # Pythia
    if hasattr(model, "gpt_neox") and hasattr(model.transformer, "layers"):
        return model.gpt_neox.layers
    raise RuntimeError("Unsupported transformer architecture")


def apply_layer_masks(model: torch.nn.Module, mask: torch.Tensor):
    blocks = find_decoder_blocks(model)
    assert mask.shape[0] == len(blocks)

    for i in range(len(blocks)):
        block = blocks[i]
        dev = next(block.parameters()).device
        block.register_buffer(
            "lesion_mask",
            mask[i].view(1, 1, -1).to(dev),
            persistent=False,
        )
        orig_forward = block.forward
        # m = mask[i].view(1, 1, -1)
        def masked_forward(*args, _orig=orig_forward, _block=block, **kwargs):
            out = _orig(*args, **kwargs)
            m = _block.lesion_mask
            if isinstance(out, tuple):
                h = out[0] * m
                return (h,) + out[1:]
            if hasattr(out, "last_hidden_state"):
                out.last_hidden_state = out.last_hidden_state * m
                return out
            return out * m

        block.forward = masked_forward
        # blocks[i] = MaskedLayer(blocks[i], mask[i])



# Argument parsing

def parse_args(argv=None):
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model-name", type=str, required=True)
    argparser.add_argument("--percentage", type=float, required=True)
    argparser.add_argument("--network", type=str, default="tom_mask",
                           choices=["tom_mask", "random", "none", "theory-of-mind"])
    argparser.add_argument("--device", type=str, default=None)
    argparser.add_argument("--seed", type=int, default=42)
    argparser.add_argument("--pooling", type=str, default="mean",
                           choices=["last-token", "mean"])
    argparser.add_argument("--localize-range", type=str, default="100-100")
    argparser.add_argument("--masks_cache", type=str, default="cache_baseline",
                           help="Cache directory for lesion masks")
    return argparser.parse_args(argv)


# Model setup with lesioning

def setup_lesioned_model(args):
    seed = args.seed
    percentage = args.percentage
    localizer_dataset, localizer_type = args.subnetwork_mask.split("|") if args.subnetwork_mask is not None else (None, None)
    if localizer_dataset == "all":
        localizer_dataset = "all-combined-selection"
    print(">>> Setting up lesioned model with args:", localizer_dataset, localizer_type, args)
    if args.device is None:
        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                device = "auto"
            else:
                device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = args.device
    print(f">>>>> Using device: {device}")

    model_name = args.model_name
    network = args.network
    pooling = args.pooling
    loc_range = args.localize_range
    cache_masks = args.masks_cache

    if ("OLMo" in model_name) or ("gemma-2" in model_name):
        dtype = torch.float32
    elif ("Llama-2" in model_name) or ("pythia" in model_name):
        dtype = torch.float16
    elif ("Llama-3" in model_name) or ("Qwen" in model_name) or ("gemma-3" in model_name) or ("Mistral" in model_name) or ("Falcon" in model_name):
        dtype = torch.bfloat16
    else:
        dtype = torch.float16

    print(f"> Running with model {model_name}")

    model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device, torch_dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        if tokenizer is not None:
            print(
                "tokenizer is changed by adding pad_token_id to the tokenizer."
            )
        if tokenizer.eos_token is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.add_special_tokens(
                {"additional_special_tokens": ["<pad>"]}
            )
            tokenizer.pad_token = "<pad>"
            model.resize_token_embeddings(len(tokenizer))
    if tokenizer.padding_side == "left":
        tokenizer.padding_side = "right"

    model_name_base = os.path.basename(model_name)
    print(f"> Running with {network} mask")

    if network in ["tom_mask"]:
        mask_path = (
            f"{model_name_base}_network=theory-of-mind_pooling={pooling}"
            f"_range={loc_range}_perc={percentage}_nunits=None_pretrained=True_dataset=tom_chat=True.npy"
        )
    elif network == "theory-of-mind":
        mask_path = (
            f"{model_name_base}_network=theory-of-mind_pooling={pooling}"
            f"_pretrained=True_suite={localizer_dataset}_percentage=1_p=0.05_mask={localizer_type}_fdr.npy"
        )
        print(" >>>>> constructed mask path:", mask_path)
    elif network == "random":
        mask_path = (
            f"{model_name_base}_network=theory-of-mind_pooling={pooling}"
            f"_range={loc_range}_perc={percentage}_nunits=None_pretrained=True"
            f"_dataset=tom_chat=True.npy"
        )
    else:
        mask_path = None

    if mask_path is not None:
        tom_mask = np.load(f"{cache_masks}/{mask_path}")
        print("Sums in the tom_mask selective mask loaded original:", tom_mask.sum())

        # if network == "random":
        #     num_layers, hidden_dim = tom_mask.shape
        #     total = num_layers * hidden_dim
        #     # TODO: check the 1 - here
        #     inv_idx = np.arange(total)[(1 - tom_mask).flatten().astype(bool)]
        #     np.random.seed(seed)
        #     rand_idx = np.random.choice(inv_idx, size=int(tom_mask.sum()), replace=False)
        #     flat = np.zeros(total)
        #     flat[rand_idx] = 1
        #     tom_mask = flat.reshape(num_layers, hidden_dim)

        # else:
        tom_mask = 1 - tom_mask
        print("Sums in the tom_mask selective mask after 1 - lang_mask:", tom_mask.sum())

        mask_tensor = torch.tensor(tom_mask, dtype=dtype, device="cpu")
        print("mask tensor device ", mask_tensor.device)
        apply_layer_masks(model, mask_tensor)

        print("Loaded mask with shape", tom_mask.shape, mask_path)

    model.eval()
    return model, tokenizer, device

# set up get_continuation_logprobs as a method that can be called on transformer models objects
def get_continuation_logprobs( # note: replication of minicons method
        model,
        encoded: torch.LongTensor,
        attention_mask: torch.Tensor = None,
        offsets: int = 0,
        **kwargs,
    ) -> torch.Tensor:
    """
    Computes the average log probability assigned to the continuation tokens (after the prefix) for each sequence.

    Args:
        input_ids: (batch_size, seq_len)
        attention_mask: (batch_size, seq_len), optional
        prefix_length: int, number of prefix tokens before continuation
    """
    # encoded, offsets = input_ids, prefix_length
    ids = [
        [i for i, am in zip(instance, attention_mask) if am != 0]
        for instance, attention_mask in zip(
            encoded["input_ids"].tolist(), encoded["attention_mask"].tolist()
        )
    ]
    ## Ignore the probabilities of the first token.
    effective_ids = [id[1:] for id in ids]
    
    with torch.no_grad():
        # print("checking for tom_mask selective mask in forward in log P code", self.tom_mask_selective_mask is None)
        outputs = model(
            **encoded
            # **kwargs,
        )
        logits = outputs.logits.detach()

    logits = logits.split([1] * len(offsets))

    ## Set up storage variables
    scores = []
    for logit, idx, offset in zip(logits, effective_ids, offsets):
        #### of the entire sequence w/o first token ####
        length = len(idx)
        print("len of effective_ids:", length)
        print("offset:", offset)

        query_ids = idx[offset:]
        logit = logit.squeeze(0)
        logprob_distribution = logit - logit.logsumexp(1).unsqueeze(1)

        actual_logprob_distribution = logprob_distribution[
            torch.arange(offset, length),
        ]

        score = actual_logprob_distribution[
            torch.arange(length - offset), query_ids
        ]
        scores.append(score)

    
    return scores

# Rest of the code for lesion evaluation metrics

def prime_text( 
    tokenizer,
    device,
    preamble,
    stimuli,
    separator=" ",
    chat=False,
    bos_token=False,
    eos_token=False,
): # NOTE: replication of minicons method
    preamble_text = [preamble] if isinstance(preamble, str) else preamble
    
    if chat == True:
        if tokenizer.chat_template is not None:
            preamble_encoded = tokenizer(
                preamble_text, add_special_tokens=False
            )["input_ids"]
        else:
            raise ValueError(
                "Chat is set to True but the model does not have a chat template."
            )
    else:
        preamble_encoded = tokenizer(preamble_text)["input_ids"]
    #### these encode the lengths of the context, corrected for presence of BOS token
    preamble_lens = []
    for preamble_tokens in preamble_encoded:
        if bos_token:
            restricted_id = float("inf")
            bos_offset = 1
        else:
            restricted_id = tokenizer.pad_token_id
            bos_offset = 0
        preamble_lens.append(
            len([token for token in preamble_tokens if token != restricted_id])
            - 1 #### probably for slicing anf indexing ##### 
            + bos_offset
        )
    #### concatenation of contexts and options WITH separator ####
    if isinstance(separator, str):
        sentences = (
            [preamble + separator + stimuli]
            if isinstance(preamble, str)
            else [p + separator + s for p, s in list(zip(preamble, stimuli))]
        )

    elif isinstance(separator, list):
        assert not isinstance(preamble, str)
        assert len(preamble) == len(separator) == len(stimuli)

        sentences = [
            p + sep + s for p, sep, s in list(zip(preamble, separator, stimuli))
        ]
    #### copy of implementation of self.encode(sentences, bos_token, eos_token, chat) ####
    def _format(text, bos, eos):
        if bos:
            text = tokenizer.bos_token + text
        if eos:
            text = text + tokenizer.eos_token
        return text

    sentences = [sentences] if isinstance(sentences, str) else sentences
    sentences = [_format(t, bos_token, eos_token) for t in sentences]
    if chat == True:
        if tokenizer.chat_template is not None:
            encoded = tokenizer(
                sentences, return_tensors="pt", padding=True, add_special_tokens=False
            )
        else:
            raise ValueError(
                "Chat is set to True but the model does not have a chat template."
            )
    else:
        encoded = tokenizer(sentences, return_tensors="pt", padding=True)

    if "token_type_ids" in encoded.keys():
        encoded.pop("token_type_ids")

    return encoded, preamble_lens

def prepare_text(
    tokenizer,
    device,
    text,
    bos_token: bool = False,
    eos_token: bool = False,
    chat: bool = False,
):
    """
    Helper for preparing inputs for scoring the entire sequence (for Blimp).
    Replication of minicons method.
    """
    def _format(text, bos, eos):
        if bos:
            text = tokenizer.bos_token + text
        if eos:
            text = text + tokenizer.eos_token
        return text

    text = [text] if isinstance(text, str) else text
    text = [_format(t, bos_token, eos_token) for t in text]
    if chat == True:
        if tokenizer.chat_template is not None:
            encoded = tokenizer(
                text, return_tensors="pt", padding=True, add_special_tokens=False
            )
        else:
            raise ValueError(
                "Chat is set to True but the model does not have a chat template."
            )
    else:
        encoded = tokenizer(text, return_tensors="pt", padding=True)

    if "token_type_ids" in encoded.keys():
        encoded.pop("token_type_ids")

    # return encoded
    offsets = [0] * len(encoded["input_ids"])
    return encoded, offsets
    
def get_lesion_item_scores(
    model,
    tokenizer,
    device,
    story: str,
    answer_options: list[str],
    dataset_name: str,
    model_name: str,
    use_chat_model_formatting: bool = False,
):
    """
    Get log probability scores for each answer option given the story context.
    Returns log probabilities for each answer option (usually two: correct and incorrect).
    
    Parameters
    ----------
    model : MistralForCausalLM
        The lesioned model
    tokenizer : AutoTokenizer
        The tokenizer for the model
    device : str
        The device to run on
    story : str
        The story/context
    answer_options : list[str]
        List of answer options to score
        
    Returns
    -------
    list
        Log probability scores for each answer option
    """
    # prime_text to get the encoding and the lengths that are expected by the new get_continuation_logprobs method
    model.eval()
    if dataset_name == "blimp":
        if ("instruct" in model_name.lower()) or ("-it" in model_name.lower()) or ("chat" in model_name.lower()):
            good_sentence = tokenizer.apply_chat_template(
                [{"role": "user", "content": story[0]}],
                tokenize = False
            )
            bad_sentence = tokenizer.apply_chat_template(
                [{"role": "user", "content":  story[1]}],
                tokenize = False
            )
            is_chat = True
        else:
            good_sentence = story[0]
            bad_sentence = story[1]
            is_chat = False
        stories = [good_sentence, bad_sentence]
        encoded, preamble_lens = prepare_text( 
            tokenizer,
            device,
            stories,
            chat=is_chat,
        )
        print("encoded blimp: ", encoded)
    else:
        # if needed, use special formatting of context and options for chat models, otherwise just pass indicator of model type
        if use_chat_model_formatting:
            story, answer_options, is_chat = special_chat_formatting(model_name, story, answer_options, dataset_name, tokenizer)
        else:
            _, _, is_chat = special_chat_formatting(model_name, story, answer_options, dataset_name, tokenizer)
        # print("chat formatted: ", story, answer_options, is_chat)
        
        stories = [story for _ in answer_options]
        encoded, preamble_lens = prime_text( 
            tokenizer,
            device,
            stories,
            answer_options,
            chat=is_chat,
        )
    if device != "auto":
        encoded.to(device)
    print("Encoded input ids shape: ", encoded["input_ids"].shape)
    print("preamble length ", preamble_lens)
    scores = get_continuation_logprobs(
        model,
        encoded=encoded,
        offsets=preamble_lens
    )
   
    mean_reduction = lambda x: x.mean(0).item()
    log_prob_scores = list(map(mean_reduction, scores))
    sum_reduction = lambda x: x.sum(0).item()
    sum_scores = list(map(sum_reduction, scores))

    return log_prob_scores, sum_scores, scores  


def run_lesion_evaluation(
        dataset,
        model,
        tokenizer,
        device,
        cfg_file,
        model_name,
        use_additional_space=True,
        use_chat_model_formatting=False,
):
    """
    Run a log probability scorer on a lesioned model. 
    This mirrors the run_logit_lens_scorer function in tomXprag/src/evaluate/metrics.py
    
    Parameters
    ----------
    dataset : DatasetBase
        The dataset to evaluate
    model : MistralForCausalLM
        The lesioned model
    tokenizer : AutoTokenizer
        The tokenizer
    device : str
        The device to run on
    cfg_file : str
        Path to the config YAML file
        
    Returns
    -------
    pd.DataFrame
        Dataset with lesion scores added
    """
    assert cfg_file is not None, "Config file must be provided!"
    with open(cfg_file, 'r') as f:
        cfg = yaml.safe_load(f)
    dataset_name = dataset.name
    
    try:
        if dataset_name.startswith("bigtom") or dataset_name.startswith("tinytom") or dataset_name.startswith("hufloyd") or dataset_name.startswith("ewok"):
            # for bigtom and tinytom datasets, we need to run scorer on two contexts
            if dataset_name.startswith("tinytom"):
                dataset_config = cfg[dataset_name.split("_")[0].split("-")[0]]
            else:
                dataset_config = cfg[dataset_name.split("_")[0]]
        elif dataset_name == "blimp":
            dataset_config = {"condition": "blimp"}
        else:
            dataset_config = cfg[dataset_name]
    except KeyError:
        raise ValueError(f"Dataset `{dataset_name}` not found in config file `{cfg_file}`!")    
    
    tomi_sapEtAl_instruction = "The following multiple choice questions is based on the following story. The question is related to Theory-of-Mind. Read the story and then answer the questions. Choose the best answer from the options provided by printing it as is without any modifications."
    
    # Iterate over the rows of the dataset and call the logit lens item scoring function
    for context in ["context1", "context2"]:
        # check if it is a 2x2 dataset
        if context in dataset_config:
            ds_config = dataset_config[context]
            # get the condition suffix for constructing the name of the results column
            col_prefix = ds_config["condition"].split("_")[-1] + "_"
        else:
            ds_config = dataset_config
            col_prefix = ""

        for i, row in tqdm(dataset.items.iterrows(), total=len(dataset.items.index)):
            # construct blimp item
            if dataset_name == "blimp":
                story = [row['sentence_good'], row['sentence_bad']]
                answer_options = []
            else:
                # Construct the context from multiple columns if needed
                if use_additional_space:
                    story = " ".join(
                        row[ds_config["context"]]
                    ).strip() + " "
                else:
                    story = " ".join(
                        row[ds_config["context"]]
                    ).strip()
                
                if "answer_options" in ds_config:
                    answer_options = [
                        row[option] for option in
                        ds_config["answer_options"]
                    ]
                    if "epitome_rm" in dataset_name:
                        correct = 0 if row[ds_config["correct_index"]] == "a" else 1
                        incorrect = 1 - correct
                        answer_options = [
                            answer_options[correct],
                            answer_options[incorrect]
                        ]
                else:
                    answer_options = [
                        row[ds_config["correct"]],
                        row[ds_config["incorrect"]]
                    ]
                
                # check if there is a prefix string
                if "answer_prefix" in ds_config:
                    if use_additional_space:
                        story = story + ds_config["answer_prefix"].strip() + " "
                    else:
                        story = story + ds_config["answer_prefix"].strip()

                if dataset_name == "tom_localizer":
                    answer_options = [str(opt) for opt in answer_options]
                # strip answer options to be sure
                answer_options = [opt.strip() for opt in answer_options]

                # format answer options for TomI
                if dataset_name == "tomi_sapEtAl_first_order":
                    row["cands"] = ast.literal_eval(row["cands"])
                    if use_additional_space:
                        print("---- using spacing like in first behavioral eval ----")
                        story = tomi_sapEtAl_instruction + "\n" + "Story: " + row["story"] + "\nQuestion: " + row["question"] + "\nOptions:\n- " + row["cands"][0] + "\n- " + row["cands"][1] + "\nAnswer: "
                    else:
                        story = tomi_sapEtAl_instruction + "\n" + "Story: " + row["story"] + "\nQuestion: " + row["question"] + "\nOptions:\n- " + row["cands"][0] + "\n- " + row["cands"][1] + "\nAnswer:"

            # Get log probability scores from model
            # print("story: ", story)
            # print("answer options: ", answer_options)

            mean_answer_logprobs, sum_answer_logprobs, raw_scores = get_lesion_item_scores(
                model,
                tokenizer,
                device,
                story,
                answer_options,
                ds_config["condition"],
                model_name,
                use_chat_model_formatting,
            )
    
            # Store the logprob scores for each answer option in the dataset
            dataset.items.at[i, f"{col_prefix}correct_lesion_mean_logprob"] = mean_answer_logprobs[0] 
            dataset.items.at[i, f"{col_prefix}incorrect_lesion_mean_logprob"] = mean_answer_logprobs[1] 
            dataset.items.at[i, f"{col_prefix}correct_lesion_sum_logprob"] = sum_answer_logprobs[0] 
            dataset.items.at[i, f"{col_prefix}incorrect_lesion_sum_logprob"] = sum_answer_logprobs[1] 
            dataset.items.at[i, f"{col_prefix}correct_lesion_logprob"] = ";".join(map(str, raw_scores[0])) 
            dataset.items.at[i, f"{col_prefix}incorrect_lesion_logprob"] = ";".join(map(str, raw_scores[1]))
            dataset.items.at[i, f"{col_prefix}scored_condition"] = ds_config["condition"]
        # only run one loop if there is no context2
        if context not in dataset_config:
            break

    
    return dataset.items

def baseline_lesion_evaluation(
        dataset,
        model,
        tokenizer,
        device,
        cfg_file,
        model_name,
        use_additional_space=True,
        use_chat_model_formatting=False,
    ):
    
    # Load the dataset
    dataset_name = dataset.name
    # iterate over the dataset
    log_prob_scores_correct = []
    log_prob_scores_incorrect = []
    for i, row in tqdm(dataset.items.iterrows(), total=len(dataset.items.index)):
        input_tokens_good = tokenizer(row["sentence_good"], return_tensors="pt").to(device)
        input_tokens_bad = tokenizer(row["sentence_bad"], return_tensors="pt").to(device)
        
        log_probs_good = get_continuation_logprobs(
            model,
            encoded=input_tokens_good["input_ids"],
            offsets=1 # NOTE: this is because we want all token scores
        )  # (batch_size,)

        log_prob_scores_correct.append(log_probs_good.squeeze(0).item())

        log_probs_bad = get_continuation_logprobs(
            model,
            encoded=input_tokens_bad["input_ids"],
            offsets=1 # NOTE: this is because we want all token scores
        )  # (batch_size,)

        log_prob_scores_incorrect.append(log_probs_bad.squeeze(0).item())

    # format the resulting df
    # and calculate if item prediction is correct based on log probs
    df["correct_lesion_logprob"] = log_prob_scores_correct
    df["incorrect_lesion_logprob"] = log_prob_scores_incorrect
    df["lesion_prediction_correct"] = df["correct_lesion_logprob"] > df["incorrect_lesion_logprob"]
    
    return df



if __name__ == "__main__":
    args = parse_args()
    model, tokenizer, device = setup_lesioned_model(args)
    print("Model setup complete. Use this module programmatically by calling setup_lesioned_model(args).")

