from typing import Literal, Optional, List
import torch as t
from nnsight import LanguageModel
from transformers import AutoTokenizer

from .modules import JumpReLUSAE, Submodule

GemmaModelSizes = Literal["2b", "9b", "27b"]

def load_gemma(
    model_size: GemmaModelSizes,
    l0 : int,
    width : int,
    custom_model_id: Optional[str] = None,
    device: Literal["cpu", "cuda"] = "cuda",
    torch_dtype: Optional[t.dtype] = None,
    layers: Optional[List[int]] = None,
):
    """
    Loads the Gemma 2 2B or 9B model and its dictionaries.

    Args:
        model_size: The size of the model to load.
        custom_model_id: The custom model ID to use.
        dictionary_types: The types of dictionaries to load.
        device: The device to load the model on.
        load_dicts: Whether to load the dictionaries.
        torch_dtype: The torch dtype to use.
        layers: The layers to load.
    """

    tokenizer = AutoTokenizer.from_pretrained(f"google/gemma-2-{model_size}")    
    model = LanguageModel(
        custom_model_id or f"google/gemma-2-{model_size}", 
        tokenizer=tokenizer,
        attn_implementation="eager", 
        device_map="auto", 
        dispatch=True, 
        torch_dtype=torch_dtype
    )

    dictionaries = {}

    if layers is None:
        layers = range(model.config.num_hidden_layers)

    for layer in layers:
        name = f".model.layers.{layer}"
        dictionaries[name] = JumpReLUSAE.from_pretrained(
            repo_id=f"google/gemma-scope-{model_size}-pt-res",
            file_name=f"layer_{layer}/width_{width}/average_l0_{l0}/params.npz",
        )
        dictionaries[name].to(device, dtype=torch_dtype)
        print(f"\rLoaded layer {layer}", end='', flush=True)
    
    submodules = []

    for name, module in model.named_modules():
        if (name in dictionaries.keys()):
            s = Submodule(
                path=name, 
                module=module, 
                dictionary=dictionaries[name]
            )
            submodules.append(s)

    return model, submodules
