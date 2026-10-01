from typing import Optional, Literal
import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import nnsight
from nnsight import LanguageModel

from minicons import scorer

from utils import get_model_family


class LM():
    """Model class for Huggingface-based LMs evaluated in our experiments."""
    def __init__(
        self, 
        model_name: str, 
        tokenizer_name: Optional[str] = None, 
        path_to_adapters: Optional[str] = None,
        model_revision: Optional[str] = None,
        **load_kwargs
    ) -> None:
        # Store basic meta data about the model.
        self.model_name = model_name
        self.path_to_adapters = path_to_adapters
        self.model_revision = model_revision
        if tokenizer_name is None:
            self.tokenizer_name = model_name
        else:
            self.tokenizer_name = tokenizer_name

        # Initialize tokenizer and model.
        print(
            f"Initializing tokenizer ({self.tokenizer_name}) "
            f"and model ({model_name})"
        )
        tokenizer = self.load_tokenizer_and_model( # model
            self.model_name, 
            self.tokenizer_name,
            self.path_to_adapters,
            self.model_revision,
            **load_kwargs
        )
        self.tokenizer = tokenizer
#        self.model = model

        # set dtype
        if ("OLMo" in self.model_name) or ("gemma-2" in self.model_name):
            dtype = torch.float32
        elif ("Llama-2" in self.model_name) or ("pythia" in self.model_name):
            dtype = torch.float16
        elif ("Llama-3" in self.model_name) or ("Qwen" in self.model_name) or ("gemma-3" in self.model_name) or ("Mistral" in self.model_name) or ("Falcon" in self.model_name):
            dtype = torch.bfloat16
        else:
            dtype = torch.float16
        # Initialize minicons scorer object to compute probabilities.
        # try to fix OLMo-2 issues
        # try:
            # try to pass chat = True
        self.scorer = scorer.IncrementalLMScorer(
            self.model_name, 
            tokenizer=tokenizer, 
            device="auto" if torch.cuda.device_count() != 1 else "cuda",
            torch_dtype=dtype,
           # chat = True if ("chat" in self.model_name.lower()) or ("instruct" in self.model_name.lower()) or ("-it" in self.model_name.lower()) else False,
        )
        # except Exception as e:
            # print("error while initializing IncrementalLMScorer", e)
            # self.scorer = scorer.IncrementalLMScorer(
            #     self.model_name,
            #     tokenizer=tokenizer,
            #     device="auto",
            # )
            
    def load_tokenizer_and_model(
        self, 
        model_name: str, 
        tokenizer_name: str, 
        path_to_adapters: Optional[str] = None,
        model_revision: Optional[str] = None,
        cache_dir: Optional[str] = None,
        **kwargs
    ):
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, 
            padding_side="left",
            cache_dir=cache_dir,
            **kwargs
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        if model_revision is not None and "OLMo" in model_name:
            from hf_olmo import OLMoForCausalLM 
            
            model = OLMoForCausalLM.from_pretrained(
                model_name, 
                revision=model_revision
            )
        # else:
        #     print("Using AutoModelForCausalLM to load the model")
        #     model = AutoModelForCausalLM.from_pretrained(
        #         model_name,
        #         device_map="auto",
        #         **kwargs
        #     )
        if path_to_adapters is not None:
            print("Loading adapters")
            #model = PeftModel.from_pretrained(
            #    model, path_to_adapters
            #)
#            model.load_adapter(path_to_adapters)
            print("loaded adapters")
            # model = model_to_merge.merge_and_unload()
            model = PeftModel.from_pretrained(
                model, 
                path_to_adapters, 
                cache_dir=cache_dir
            )
        return tokenizer #, model
    
def tokenize_batch(
    tokenizer,
    examples,
    col: str = "text",
):
    """
    Utility function for tokenizing text.
    """
    tokenized = tokenizer(
        examples[col], 
        padding="longest", 
        truncation=True,
    )
    return tokenized



class nnsightLM():
    """Wrapper for an nnsight LanguageModel object"""
    def __init__ (
            self, 
            model_name: str
    ) -> None:
        # basic metadata
        self.model_name = model_name
        self.model_family = get_model_family(model_name)

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, device="auto")
        if tokenizer.pad_token is None:
            print("No pad token found; setting pad token to eos token")
            tokenizer.pad_token = tokenizer.eos_token
        self.tokenizer = tokenizer

        # load nnsight wrapped model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto"
        )
        self.model = LanguageModel(
            model,
            tokenizer=tokenizer,
        )
        print(f"Loaded model {model_name} with nnsight wrapper, on device {self.model.device}")
        # unified refrences to model internals across different model families
        if self.model_family in ["llama", "olmo", "gemma", "falcon"]:
            self.model.model = self.model.model
            self.model.layers = self.model.model.layers
            self.model.layer_norm = self.model.model.norm
            self.model.lm_head = self.model.lm_head
        elif self.model_family == "gpt":
            self.model.model = self.model.transformer
            self.model.layers = self.model.transformer.h
            self.model.layer_norm = self.model.transformer.ln_f
            self.model.lm_head = self.model.lm_head
        elif self.model_family == "mamba":
            self.model.layers = self.model.backbone.layers
            self.model.layer_norm = self.model.backbone.norm_f
            self.model.lm_head = self.model.lm_head
        else:
            raise ValueError(f"Unsupported model family: {self.model_family}")
        
                      
            


