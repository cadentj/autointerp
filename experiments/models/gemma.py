from typing import Optional, List
import torch as t
import os
import json
from nnsight import LanguageModel

from .modules import JumpReLUSAE, Submodule

# GemmaModelSizes = Literal["2b", "9b", "27b"]

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

def load_gemma(
    # model_size: GemmaModelSizes,
    # l0 : int,
    # width : int,
    torch_dtype: Optional[t.dtype] = t.bfloat16,
    layers: Optional[List[int]] = None,
):
    path = os.path.join(CURRENT_DIR, "gemmascope_16k_canonical.json")
    with open(path, "r") as f:
        gemma_config = json.load(f)

    model = LanguageModel(
        "google/gemma-2-2b",
        attn_implementation="eager", 
        device_map="auto", 
        dispatch=True, 
        torch_dtype=torch_dtype
    )

    if layers is None:
        layers = range(model.config.num_hidden_layers)

    layer_modules = model.model.layers
    submodules = []

    for layer, config in zip(layers, gemma_config["layers"]):
        name = f".model.layers.{layer}"

        dictionary = JumpReLUSAE.from_pretrained(
            repo_id="google/gemma-scope-2b-pt-res",
            file_name=config["path"] + "/params.npz",
        )

        dictionary = dictionary.to("cuda", dtype=torch_dtype)
        
        s = Submodule(
            path=name, 
            module=layer_modules[layer], 
            dictionary=dictionary
        )
        submodules.append(s)

        print(f"\rLoaded layer {layer}", end='', flush=True)

    print()
    
    return model, submodules
