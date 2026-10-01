model_name_map = {
    "gpt2": "GPT2-Small",
    "gpt2-medium": "GPT2-Med",
    "gpt2-large": "GPT2-Large",
    "gpt2-xl": "GPT2-XL",
    "Llama-2-7b-hf": "LLaMA-2-7b",
    "Llama-2-7b-hf": "LLaMA-2-7b",
    "Llama-2-7b-chat-hf": "LLaMA-2-7b-Instruct",
    "Llama-2-13b-hf": "LLaMA-2-13b",
    "Llama-2-13b-chat-hf": "LLaMA-2-13b-Instruct",
    "Llama-2-70b-hf": "LLaMA-2-70b",
    "Llama-2-70b-chat-hf": "LLaMA-2-70b-Instruct",
    "Llama-3.1-8B": "Llama-3.1-8B",
    "Llama-3.1-8B-Instruct": "LLaMA-3.1-8B-Instruct",
    "Llama-3.2-3B-Instruct": "LLaMA-3.2-3B-Instruct",
    "Llama-3.2-3B": "LLaMA-3.2-3B",
    "Llama-2-70b-chat-hf": "Llama-2-70b-Instruct",
    "Llama-3.1-70B": "Llama-3.1-70B",
    "Llama-3.3-70B-Instruct": "Llama-3.3-70B-Instruct",
    "Phi-3.5-mini-instruct": "Phi-3.5-Mini-Instruct",
    "gemma-1.1-7b-it": "Gemma-1.1-7B-Instruct",
    "gemma-2b": "Gemma-2B",
    "gemma-7b": "Gemma-7B",
    "gemma-2-9b": "Gemma-2-9B",
    "gemma-2-9b-it": "Gemma-2-9B-Instruct",
    "gemma-2-27b": "Gemma-2-27B",
    "gemma-2-27b-it": "Gemma-2-27B-Instruct",
    "falcon-7b": "Falcon-7B",
    "falcon-7b-instruct": "Falcon-7B-Instruct",
    "Falcon3-10B-Instruct": "Falcon-10B-Instruct",
    "Falcon3-10B-Base": "Falcon-10B",
    "Falcon3-7B-Instruct": "Falcon-7B-Instruct",
    "Falcon3-7B-Base": "Falcon-7B",
    "Mistral-7B-Instruct-v0.3": "Mistral-7B-Instruct",
    "Mistral-7B-v0.3": "Mistral-7B",
    "pythia-6.9b-deduped": "pythia-6.9b",
    "pythia-12b-deduped": "pythia-12b",
    "Qwen2.5-7B-Instruct": "Qwen2.5-7B-Instruct",
    "Qwen2.5-7B": "Qwen2.5-7B",
    "Qwen2.5-14B-Instruct": "Qwen2.5-14B-Instruct",
    "Qwen2.5-14B": "Qwen2.5-14B",
    "Qwen2.5-32B-Instruct": "Qwen2.5-32B-Instruct",
    "Qwen2.5-32B": "Qwen2.5-32B",
    "Qwen2.5-72B-Instruct": "Qwen2.5-72B-Instruct",
    "Qwen2.5-72B": "Qwen2.5-72B",
    "OLMo-7B-0724-hf": "OLMo-7B",
    "OLMo-7B-0724-Instruct-hf": "OLMo-7B-Instruct"

    
}

def get_num_blocks(model_name):
    return {
        "gpt2": 12,
        "gpt2-medium": 24,
        "gpt2-large": 36,
        "gpt2-xl": 48,
        "Llama-2-7b-hf": 32,
        "Llama-2-7b-chat-hf": 32,
        "vicuna-7b-v1.3": 32,
        "Llama-2-13b-hf": 40,
        "Llama-2-13b-chat-hf": 40,
        "Llama-3.1-8B-Instruct": 32,
        "Llama-3.1-8B": 32,
        "Llama-3.2-3B": 28,
        "Llama-3.2-3B-Instruct": 28,
        "Llama-3.1-70B": 80,
        "Llama-3.3-70B-Instruct": 80,
        "Llama-2-70b-hf": 80,
        "Llama-2-70b-chat-hf": 80,

        "Phi-3.5-mini-instruct": 32,

        "falcon-7b": 32,
        "falcon-7b-instruct": 32,

        "Mistral-7B-v0.3": 32,
        "Mistral-7B-Instruct-v0.3": 32,

        "vicuna-13b-v1.3": 40,

        "gemma-2b": 18,
        "gemma-2b-it": 18,
        "gemma-2-2b": 26,
        "gemma-2-9b-it": 42,
        "gemma-2-9b": 42,
        "gemma-2-27b-it": 46,
        "gemma-2-27b": 46,

        "gemma-1.1-2b-it": 18,
        "gemma-7b": 28,
        "gemma-1.1-7b-it": 28,
        "gemma-3-12b-pt": 48,
        "gemma-3-12b-it": 48,
        "gemma-3-27b-pt": 62,
        "gemma-3-27b-it": 62, # num_hidden_layers in the config

        "Qwen2.5-0.5B": 24,
        "Qwen2.5-0.5B-Instruct": 24,
        "Qwen2.5-1.5B": 28,
        "Qwen2.5-1.5B-Instruct": 28,
        "Qwen2.5-3B": 36,
        "Qwen2.5-3B-Instruct": 36,
        "Qwen2.5-7B": 28,
        "Qwen2.5-7B-Instruct": 28,
        "Qwen2.5-14B": 48,
        "Qwen2.5-14B-Instruct": 48,
        "Qwen2.5-32B": 64,
        "Qwen2.5-32B-Instruct": 64,
        "Qwen2.5-72B": 80,
        "Qwen2.5-72B-Instruct": 80,

        "OLMo-1B-0724-hf": 16,
        "OLMo-7B-0724-hf": 32,
        "OLMo-7B-0724-Instruct-hf": 32,
        "OLMo-2-1124-13B": 40,

        "pythia-1b-deduped": 16,
        "pythia-1.4b-deduped": 24,
        "pythia-2.8b-deduped": 32,
        "pythia-6.9b-deduped": 32,
        "pythia-12b-deduped": 36,

        # add falcon
        "Falcon3-7B-Base": 28, 
        "Falcon3-7B-Instruct": 28, 
        "Falcon3-10B-Base": 40,
        "Falcon3-10B-Instruct": 40,

        "OLMo-2-0425-1B-Instruct": 16,
        "Falcon3-1B-Instruct": 18,
        "Falcon3-1B-Base": 18,

        "Llama-3.2-1B-Instruct": 16


    }[model_name]

def get_hidden_dim(model_name): # hidden_size in the config
    return {
        "gpt2": 768,
        "gpt2-medium": 1024,
        "gpt2-large": 1280,
        "gpt2-xl": 1600,

        "Llama-2-7b-hf": 4096,
        "Llama-2-7b-chat-hf": 4096,

        "Llama-2-13b-hf": 5120,
        "Llama-2-13b-chat-hf": 5120,

        "Llama-3.1-8B-Instruct": 4096,
        "Llama-3.1-8B": 4096,

        "Llama-3.2-3B": 3072,
        "Llama-3.2-3B-Instruct": 3072,

        "Llama-3.1-70B": 8192,
        "Llama-3.3-70B-Instruct": 8192,
        "Llama-2-70b-hf": 8192,
        "Llama-2-70b-chat-hf": 8192,

        "Phi-3.5-mini-instruct": 3072,

        "gemma-1.1-7b-it": 3072,
        
        "gemma-2-2b": 2304,
        "gemma-2-9b": 3584,
        "gemma-2-9b-it": 3584,

        "gemma-2-27b-it": 4608,
        "gemma-2-27b": 4608,

        "gemma-2b": 2048,
        "gemma-2b-it": 2048,
        "gemma-7b": 3072,

        "gemma-3-12b-pt": 3840,
        "gemma-3-12b-it": 3840,
        "gemma-3-27b-pt": 5376,
        "gemma-3-27b-it": 5376,

        "falcon-7b": 4544,
        "falcon-7b-instruct": 4544,

        "Mistral-7B-v0.3": 4096,
        "Mistral-7B-Instruct-v0.3": 4096,

        "Qwen2.5-0.5B": 896,
        "Qwen2.5-0.5B-Instruct": 896,
        "Qwen2.5-1.5B": 1536,
        "Qwen2.5-1.5B-Instruct": 1536,
        "Qwen2.5-3B": 2048,
        "Qwen2.5-3B-Instruct": 2048,
        "Qwen2.5-7B": 3584,
        "Qwen2.5-7B-Instruct": 3584,
        "Qwen2.5-14B": 5120,
        "Qwen2.5-14B-Instruct": 5120,
        "Qwen2.5-32B": 5120,
        "Qwen2.5-32B-Instruct": 5120,
        "Qwen2.5-72B": 8192,
        "Qwen2.5-72B-Instruct": 8192,

        "OLMo-1B-0724-hf": 2048,
        "OLMo-7B-0724-hf": 4096,
        "OLMo-7B-0724-Instruct-hf": 4096,
        "OLMo-2-1124-13B": 5120,

        "pythia-1b-deduped": 2048,
        "pythia-1.4b-deduped": 2048,
        "pythia-2.8b-deduped": 2560,
        "pythia-6.9b-deduped": 4096,
        "pythia-12b-deduped": 5120,

        "Falcon3-7B-Base": 3072, 
        "Falcon3-7B-Instruct": 3072, 
        "Falcon3-10B-Base": 3072,
        "Falcon3-10B-Instruct": 3072,

        "OLMo-2-0425-1B-Instruct": 2048,
        "Falcon3-1B-Instruct": 2048,
        "Falcon3-1B-Base": 2048,

        "Llama-3.2-1B-Instruct": 2048

    }[model_name]


def get_layer_names(model_name):
    num_blocks = get_num_blocks(model_name)

    if "gpt2" in model_name or "falcon" in model_name:
        return [f'transformer.h.{block}' 
            for block in range(num_blocks) 
            # for layer_desc in ['ln_1', 'attn', 'ln_2', 'mlp']
        ]
    elif "Llama" in model_name or "gemma" in model_name or "Phi" in model_name or "Mistral" in model_name or "OLMo" in model_name or "Qwen" in model_name or "Falcon" in model_name:
        return [f'model.layers.{layer_num}' 
            for layer_num in range(num_blocks) 
            # for layer_desc in ["input_layernorm", "self_attn", "post_attention_layernorm", "mlp"]
        ]  
    elif "pythia" in model_name:
        return [f'gpt_neox.layers.{layer_num}' 
            for layer_num in range(num_blocks) 
            # for layer_desc in ["input_layernorm", "self_attn", "post_attention_layernorm", "mlp"]
        ]  
    else:
        raise ValueError(f"{model_name} not supported currently!")
