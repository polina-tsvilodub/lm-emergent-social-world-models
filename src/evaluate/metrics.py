from tqdm import tqdm
import yaml
import numpy as np
import torch
import ast
from typing import Literal
from utils import special_chat_formatting

def score_item(
    model, 
    context: str, 
    answer_options: list[str], 
    correct_index: int = 0,
    dataset: str = "",
    **kwargs
) -> dict:
    """
    Helper function for getting scores corresponding to correct and incorrect
    answer options for a particular item.

    `answer_options` should only have two options: correct and incorrect
    """
    assert len(answer_options) == 2
    incorrect_index = 1 - correct_index
    print("######### dataset ", dataset, " #########")
    context, answer_options, is_chat = special_chat_formatting(model, context, answer_options, dataset)
    print("Context after special formatting check: ", context, answer_options)
    kwargs.update({"chat": is_chat})
    print("updated kwargs ", kwargs)
    # Compute conditional sum logprobs.
    prefixes = [context for _ in answer_options]
    all_scores = model.scorer.conditional_score(
        prefixes,
        answer_options,
        separator=" ",
        reduction=lambda x: (x.mean(0).item(), x.sum(0).item()),
        **kwargs
    )
    mean_logprobs, sum_logprobs = zip(*all_scores)

    result = {}
    for metric in ["mean", "sum"]:
        # Look at mean logprobs (across tokens) or sum logprobs.
        scores = mean_logprobs if metric == "mean" else sum_logprobs

        # Label the outputs according to the indices.
        correct_score = scores[correct_index]
        incorrect_score = scores[incorrect_index]

        # Update this row with model outputs.
        result.update({
            f"correct_option_{metric}_logprob": correct_score,
            f"incorrect_option_{metric}_logprob": incorrect_score,
            f"model_correct_{metric}_logprob": (correct_score > incorrect_score)
        })
    return result

def run_scorer(dataset, model, dataset_config, dataset_name):
    """
    Wrapper for calling the scorer on a dataset with respective specific formatting.
    """
    tomi_sapEtAl_instruction = "The following multiple choice questions is based on the following story. The question is related to Theory-of-Mind. Read the story and then answer the questions. Choose the best answer from the options provided by printing it as is without any modifications."
    
    for context in ["context1", "context2"]:
        #  check if it is a 2x2 dataset
        if context in dataset_config:
            ds_config = dataset_config[context]
            # get the condition suffix for constructing the name of the results column
            col_prefix = ds_config["condition"].split("_")[-1] + "_"
        else:
            ds_config = dataset_config
            col_prefix = ""

        for i, row in tqdm(dataset.iterrows(), total=len(dataset.index)):
            if dataset_name == "blimp":
                print("Processing BLIMP dataset row")
                # compare scores of entire sentences
                if ("instruct" in model.model_name.lower()) or ("-it" in model.model_name.lower()) or ("chat" in model.model_name.lower()):
                    good_sentence = model.tokenizer.apply_chat_template(
                        [{"role": "user", "content": row['sentence_good']}],
                        tokenize = False
                    )
                    bad_sentence = model.tokenizer.apply_chat_template(
                        [{"role": "user", "content":  row['sentence_bad']}],
                        tokenize = False
                    )
                    is_chat = True
                else:
                    good_sentence = row['sentence_good']
                    bad_sentence = row['sentence_bad']
                    is_chat = False
                print("Blimp good sentence, bad sentence after formatting: ", is_chat, good_sentence, bad_sentence)
                # set reduction to none to get per-token logprobs and both sum and average them for results
                scores = model.scorer.sequence_score(
                    [good_sentence, bad_sentence],
                    reduction=lambda x: x,
                    chat=is_chat,
                )
                res = {}
                for metric in ["mean", "sum"]:
                    if metric == "mean":
                        reduction = lambda x: x.mean(0).item()
                    else:
                        reduction = lambda x: x.sum(0).item()
                    scores_agg = list(map(reduction, scores))
                    # Label the outputs according to the indices.
                    correct_score = scores_agg[0]
                    incorrect_score = scores_agg[1]

                    # Update this row with model outputs.
                    res.update({
                        f"correct_option_{metric}_logprob": correct_score,
                        f"incorrect_option_{metric}_logprob": incorrect_score,
                        f"model_correct_{metric}_logprob": (correct_score > incorrect_score)
                    })
            # do conditional scoring
            else:
                # Construct the context from multiple columns if needed.
                story = " ".join(
                    row[ds_config["context"]]
                ).strip() + " "
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
                    
                # capture the two alternative items case
                else:
                    answer_options = [
                        row[ds_config["correct"]],
                        row[ds_config["incorrect"]]
                    ]
                
                # check if there is a prefix string
                if "answer_prefix" in ds_config:
                    story = story + ds_config["answer_prefix"].strip() + " "
                if dataset_name == "tom_localizer":
                    answer_options = [str(opt) for opt in answer_options]
                # strip answer options to be sure
                answer_options = [opt.strip() for opt in answer_options]
                if dataset_name == "tomi_sapEtAl_first_order":
                    row["cands"] = ast.literal_eval(row["cands"])
                    story = tomi_sapEtAl_instruction + "\n" + "Story: " + row["story"] + "\nQuestion: " + row["question"] + "\nOptions:\n- " + row["cands"][0] + "\n- " + row["cands"][1] + "\nAnswer: "
            
                # Get scores from model.
                print("Story: ", story)
                res = score_item(
                    model,
                    story,
                    answer_options,
                    correct_index=0,
                    dataset=ds_config["condition"],
                )

                # record condition
                res["scored_condition"] = ds_config["condition"]

            # Update this row with model outputs.
            for k, v in res.items():
                dataset.loc[i, f"{col_prefix}{k}"] = v
        # only run one loop if there is no context2
        if context not in dataset_config:
            break

    return dataset

def evaluate(dataset, model, cfg_file = "src/evaluate/dataset_configs.yaml"):
    """
    Wrapper function for evaluation.
    """
    # read yaml config with information on dataset sepcific processing
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
            dataset_config = {}
        else:
            dataset_config = cfg[dataset_name]
    except KeyError:
        raise ValueError(f"Dataset `{dataset_name}` not found in config file `{cfg_file}`!")    

    df = run_scorer(dataset.items, model, dataset_config, dataset_name)
    df["model"] = model.model_name

    return df


def get_logit_lens_item_scores(
    model, 
    context: str, 
    answer_options: list[str],
    reduction: Literal["sum", "mean", "first"] = "mean" 
):
    """
    Get logit lens item scores for a specific context and answer options.
    
    Arguments:
    ---------
        model: LM object as needed for logit lens, instantiated in run_logit_lens.py (#AP: an nnsight LanguageModel object)
        context: the context string to condition on
        answer_options: list of answer options to score. ASSUMPTION: the first answer option is the correct one, the second is the incorrect one.
    Returns:
    -------
        correct_logprob: torch.Tensor of shape (n_layers,) with the logit lens logprobs for the correct answer option
        incorrect_logprob: torch.Tensor of shape (n_layers,) with the logit lens logprobs for the incorrect answer option
    """
    def _get_logprob(answer: str) -> torch.Tensor:
        with model.model.trace(context+answer):
            all_layer_hiddens = [
                model.model.layers[i].output[0][0]
                for i in range(len(model.model.layers))
            ]
            hiddens = torch.stack(all_layer_hiddens, dim=0)      # (n_layers, max_seq_len, hidden_size)
            rms_out = model.model.layer_norm(hiddens).float()     # (n_layers, max_seq_len, hidden_size)
            logits = model.model.lm_head(rms_out).float()          # (n_layers, max_seq_len, vocab_size)
            logprobs = logits.log_softmax(dim=-1).save()                  # (n_layers, max_seq_len, vocab_size)
            
        context_ids = model.tokenizer(context, return_tensors="pt")["input_ids"][0].to(model.model.device)
        text_ids = model.tokenizer(context+answer, return_tensors="pt")["input_ids"][0].to(model.model.device)
        continuation_ids = text_ids[len(context_ids):]            # slice of text_ids to avoid issues with dyamic tokenization
        n_continuation_tokens = len(continuation_ids)
        continuation_logprobs = logprobs[:, -n_continuation_tokens-1:-1, :] # (n_layers, T, vocab_size)
    
        all_layer_logprobs = continuation_logprobs.gather(
                        dim=2,
                        index=continuation_ids.view(1, -1, 1).expand(len(all_layer_hiddens), -1, 1)
                    ).squeeze(2).cpu()            # (T, L)
        if reduction == "sum":
            alllogprobs = torch.sum(all_layer_logprobs, axis=-1)
        elif reduction == "mean":
            alllogprobs = torch.mean(all_layer_logprobs, axis=-1)
        elif reduction == "first":
            alllogprobs = all_layer_logprobs[0]
        else:
            raise ValueError("`reduction` should be 'sum', 'mean' or 'first'.")
        return alllogprobs
    
    answer_logprobs = []
    for answer in answer_options:
        answer_logprob = _get_logprob(answer)
        answer_logprobs.append(answer_logprob)
        
    return answer_logprobs 

def run_logit_lens_evaluation(
        dataset,
        model,
        cfg_file="src/evaluate/dataset_configs.yaml",
    ):
    """
    Run logit lens evaluation on a dataset. Will largely mirror the evaluate + run_scorer function above, with the dataset-specific processing of columns like above.
    Arguments:
    ---------
        dataset: CSVDataset object implemented in data.py
        model: LM object as needed for logit lens, instantiated in run_logit_lens.py
        cfg_file: path to config file with dataset specific information
    Returns:
    -------
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
        else:
            dataset_config = cfg[dataset_name]
    except KeyError:
        raise ValueError(f"Dataset `{dataset_name}` not found in config file `{cfg_file}`!")    
    
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
            # Construct the context from multiple columns if needed
            story = " ".join(
                row[ds_config["context"]]
            ).strip() + " "
            
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
                story = story + ds_config["answer_prefix"].strip() + " "

            # strip answer options to be sure
            answer_options = [opt.strip() for opt in answer_options]
            # Get log probability scores from model
            print("story: ", story)
            # Get logit lens scores from model
            answer_logprobs = get_logit_lens_item_scores(
                model,
                story,
                answer_options
            )
            correct_logprob = answer_logprobs[0] 
            incorrect_logprob = answer_logprobs[1]
            # Store the logit lens scores for each layer
            for layer in range(correct_logprob.shape[-1]):
                dataset.items.at[i, f"{col_prefix}correct_logit_lens_logprob_layer_{layer}"] = correct_logprob[layer].detach().cpu().item()
                dataset.items.at[i, f"{col_prefix}incorrect_logit_lens_logprob_layer_{layer}"] = incorrect_logprob[layer].detach().cpu().item()
            dataset.items.at[i, f"{col_prefix}scored_condition"] = ds_config["condition"]

        # only run one loop if there is no context2
        if context not in dataset_config:
            break
    
    dataset.items["model"] = model.model_name
    
    return dataset.items
