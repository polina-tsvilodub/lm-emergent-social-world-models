DATASET_CONTINUATION = ["blimp", "tom_localizer", "epitome_rm", "epitome_fb_fb", "epitome_fb_tb", "epitome_si", "ewok", "simpletom_behavior", "bigtom", "tinytom", "hufloyd", "ludwig", "entity_tracking", "verbal_analogies"]
DATASET_ASSISTANT = ["tomi_sapEtAl_first_order", "triangle", "social_iqa", "fauxpas_eai_q1", "fauxpas_eai_q4", "tomi_tb", "tomi_fb", "tomi_sofb", "opentom_location_cg_fo", "opentom_location_cg_so", "opentom_location_cg_fo_long", "opentom_location_cg_so_long", "opentom_location_fg_fo_new", "opentom_location_fg_so_new", "opentom_multihop_fo", "opentom_multihop_so", "opentom_attitude", "pub_deictic_qa", "pub_sarcasm", "pub_agreement", "pub_indirectness_interpretation", "pub_indirectness_classification", "imppres_implicature", "imppres_presupposition", "sarcv2", "emobench_understanding_emotion", "emobench_understanding_cause", "emobench_application", "snli", "story_analogies"]

import torch

def get_gpu_memory():
    """Return GPU memory usage in GB"""
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / 1024**3
        memory_reserved = torch.cuda.memory_reserved() / 1024**3
        return memory_allocated, memory_reserved
    elif torch.backends.mps.is_available():
        # MPS doesn't have direct memory querying, but we can get current allocated
        memory_allocated = torch.mps.current_allocated_memory() / 1024**3
        # MPS doesn't have concept of reserved memory like CUDA
        return memory_allocated, 0
    return 0, 0

def get_model_family(model_name):
    model_name = model_name.lower()
    families = ["llama", "olmo", "gemma", "gpt", "falcon", "mamba"]
    for family in families:
        if family in model_name:
            return family
        else:
            continue
    raise ValueError(f"Unrecognized model family for {model_name}")

def get_file_safe_model_name(model: str) -> str:
    """
    Returns a file-safe version of a Huggingface model identifier by
    only keeping the model name after a forward slash (/).
    Example: meta-llama/Llama-2-7b-hf --> Llama-2-7b-hf
    """
    safe_model_name = model.split("/")[-1] if "/" in model else model
    return safe_model_name

def move_answer_prefix_to_assistant(text):
    """
    Helper for moving answer prefixes to assistant role in chat formatting.
    """
    prefixes = ["The answer is:", "Answer:", "Speaker 2", "The response is:",]
    content = ""
    for prefix in prefixes:
        if prefix in text:
            text = text.strip().replace(prefix, "").strip()
            content = prefix

    return text, content

def special_chat_formatting(model: str, context: str, answer_options: list, dataset: str, tokenizer = None) -> bool:
    """
    Returns True if special formatting is needed for the given model name.
    Currently, models with 'chat' or 'instruct' in their names are flagged.
    """
    def llama_formatting(context, answer_options):
        """
        Formatting specific to Llama-2 chat models.
        """
        formatted_prompt1 = f"<s>[INST]\n{context + answer_options[0]} [/INST]"
        formatted_prompt2 = f"<s>[INST]\n{context + answer_options[1]} [/INST]"
        return formatted_prompt1, formatted_prompt2

    def gemma_formatting(context, answer_options, dataset):
        """
        Formatting specific to Gemma-3 instruct (-it) models.
        """
        if dataset in DATASET_CONTINUATION:
            formatted_prompt1 = tok.apply_chat_template(
                [{"role": "user", "content": [{"type": "text", "text": context + answer_options[0]}]}],
                tokenize = False
            )
            formatted_prompt2 = tok.apply_chat_template(
                [{"role": "user", "content": [{"type": "text", "text": context + answer_options[1]}]}],
                tokenize = False
            )
            return formatted_prompt1, formatted_prompt2
        elif dataset in DATASET_ASSISTANT:
            context, content = move_answer_prefix_to_assistant(context)
            formatted_prompt = tok.apply_chat_template(
                [{"role": "user", "content": [{"type": "text", "text": context}]},
                {"role": "assistant", "content": [{"type": "text", "text": content}]}],
                tokenize = False,
                continue_final_message=True
            )
            return formatted_prompt
        else:
            raise NotImplementedError(f"Gemma-3 formatting not implemented for dataset {dataset}")
        
    def continuation_formatting(context, answer_options):
        """
        Formatting specific to CONTINUATION datasets.
        """
        print("Using continuation formatting")
        formatted_prompt1 = tok.apply_chat_template(
            [{"role": "user", "content": context + answer_options[0]}],
            tokenize = False
        )
        formatted_prompt2 = tok.apply_chat_template(
            [{"role": "user", "content": context + answer_options[1]}],
            tokenize = False
        )
    
        return formatted_prompt1, formatted_prompt2
    
    def assistant_formatting(context):
        """
        Formatting specific to ASSISTANT datasets (those with instructions or 'The answer is' prefixes or similar).
        """
        print("Using assistant formatting")
        # handle prefix
        context, content = move_answer_prefix_to_assistant(context)
        formatted_prompt = tok.apply_chat_template(
            [
                {"role": "user", "content": context},
                {"role": "assistant", "content": content}
            ], 
            tokenize=False, 
            continue_final_message=True
        )

        return formatted_prompt
    try:
        model_name = model.model_name
    except: 
        model_name = model

    is_chat = False
    tok = tokenizer if tokenizer is not None else model.tokenizer
    
    if dataset.startswith("bigtom") or dataset.startswith("tinytom") or dataset.startswith("hufloyd") or dataset.startswith("ewok"):
        dataset = dataset.split("_")[0].replace("-v3", "").replace("-v4", "")
    # same processing for any dataset
    if (model_name == "meta-llama/Llama-2-7b-chat-hf") or (model_name == "meta-llama/Llama-2-13b-chat-hf") or (model_name == "meta-llama/Llama-2-70b-chat-hf"):
        print("#### Using special Llama-2 chat formatting ###")
        # manual processing required:
        formatted_prompt1, formatted_prompt2 = llama_formatting(context, answer_options)
        formatted_option1 = formatted_prompt1[formatted_prompt1.rindex(answer_options[0]):]
        formatted_option2 = formatted_prompt2[formatted_prompt2.rindex(answer_options[1]):]
        context = formatted_prompt1[:formatted_prompt1.rindex(answer_options[0])]
        answer_options = [formatted_option1, formatted_option2]
        is_chat = True

    else:
        if ("instruct" in model_name.lower()) or ("-it" in model_name.lower()) or ("chat" in model_name.lower()):
            print("#### Using special chat/instruct formatting ###")
            is_chat = True
            if dataset in DATASET_CONTINUATION:
                if ("gemma-3" in model_name) and ("-it" in model_name):
                    formatted_prompt1, formatted_prompt2 = gemma_formatting(context, answer_options, dataset)
                else:
                    formatted_prompt1, formatted_prompt2 = continuation_formatting(context, answer_options)
                formatted_option1 = formatted_prompt1[formatted_prompt1.rindex(answer_options[0]):]
                formatted_option2 = formatted_prompt2[formatted_prompt2.rindex(answer_options[1]):]
                context = formatted_prompt1[:formatted_prompt1.rindex(answer_options[0])]
                answer_options = [formatted_option1, formatted_option2]
                
            elif dataset in DATASET_ASSISTANT:
                if ("gemma-3" in model_name) and ("-it" in model_name):
                    context = gemma_formatting(context, answer_options, dataset)
                else:
                    context = assistant_formatting(context)
            else:
                raise NotImplementedError(f"Special formatting not implemented for dataset {dataset}")
    
    return context, answer_options, is_chat
