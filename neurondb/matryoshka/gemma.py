from typing import Literal, Optional
import torch as t
from nnsight import LanguageModel
from transformers import AutoTokenizer
from sae_lens import SAE
from typing import List, Optional

from .utils import Submodule, load_matryoshka_sae

def load_gemma(
    model_size: Literal["2b", "9b"] = "2b",
    custom_model_id: Optional[str] = None,
    dictionary_types: Literal["resid", "attn", "mlp", "all", "matryoshka"] = "resid",
    device: str = "cuda",
    load_dicts: bool = True,
    torch_dtype: Optional[t.dtype] = None,
    layers: Optional[List[int]] = None,
):
    """
    Loads the Gemma 2 2B or 9B model and its dictionaries.
    Supports multi-GPU usage when device='auto'.
    """

    # Get list of available GPU devices
    num_gpus = t.cuda.device_count()
    
    if device == "auto":
        # Create a custom device map that evenly distributes layers
        device_map = {
            "model.embed_tokens": 0,
            "model.norm": num_gpus - 1,  # Put final norm layer on last GPU
            "lm_head": 0,  # Output layer on first GPU
        }
    else:
        device_map = {
            "model.embed_tokens": device,
            "model.norm": device,  # Put final norm layer on last GPU
            "lm_head": device,  # Output layer on first GPU
        }
    
    # Calculate number of layers and layers per GPU
    if model_size == "2b":
        num_layers = 26
    else:
        num_layers = 42
        
    layers_per_gpu = num_layers // num_gpus
    remainder = num_layers % num_gpus
    
    if device == "auto":
        # Distribute layers evenly
        current_layer = 0
        for gpu in range(num_gpus):
            # Add extra layer to early GPUs if division isn't even
            extra_layer = 1 if gpu < remainder else 0
            gpu_layers = layers_per_gpu + extra_layer
            
            for i in range(gpu_layers):
                device_map[f"model.layers.{current_layer}"] = gpu
                current_layer += 1
    else:
        for i in range(num_layers):
            device_map[f"model.layers.{i}"] = device
        
    tokenizer = AutoTokenizer.from_pretrained(f"google/gemma-2-{model_size}")    
    model = LanguageModel(
        custom_model_id or f"google/gemma-2-{model_size}", 
        tokenizer=tokenizer,
        attn_implementation="eager",
        device_map=device_map,  # Use our custom device map
        dispatch=True,
        torch_dtype=torch_dtype
    )

    if not load_dicts:
        return model, []

    dictionaries = {}

    if layers is None:
        layers = range(model.config.num_hidden_layers)

    # Get the device map from the model after it's loaded
    layer_device_map = model.hf_device_map

    for i, layer in enumerate(layers):
        # Get the device for this layer from the device map
        layer_name = f"model.layers.{layer}"
        # Default to the specified device if layer not found in device map
        layer_device = layer_device_map.get(layer_name, device)

        if dictionary_types == "resid" or dictionary_types == "all":
            name = f"model.layers.{layer}"
            dictionaries[name] = SAE.from_pretrained(
                release=f"gemma-scope-{model_size}-pt-res-canonical",
                sae_id=f"layer_{layer}/width_16k/canonical",
                device=layer_device,  # Use same device as the layer
            )[0]
            if torch_dtype:
                dictionaries[name] = dictionaries[name].to(torch_dtype)

        if dictionary_types == "attn" or dictionary_types == "all":
            name = f"model.layers.{layer}.self_attn"
            attn_device = layer_device_map.get(name, layer_device)  # Fall back to layer device
            dictionaries[name] = SAE.from_pretrained(
                release=f"gemma-scope-{model_size}-pt-attn-canonical",
                sae_id=f"layer_{layer}/width_16k/canonical",
                device=attn_device,
            )[0].to(torch_dtype)

        if dictionary_types == "mlp" or dictionary_types == "all":
            name = f"model.layers.{layer}.mlp"
            mlp_device = layer_device_map.get(name, layer_device)  # Fall back to layer device
            dictionaries[name] = SAE.from_pretrained(
                release=f"gemma-scope-{model_size}-pt-mlp-canonical",
                sae_id=f"layer_{layer}/width_16k/canonical",
                device=mlp_device,
            )[0].to(torch_dtype)

        if dictionary_types == "matryoshka":
            name = f"model.layers.{layer}"
            dictionaries[name] = load_matryoshka_sae(layer, layer_device)
            if torch_dtype:
                dictionaries[name] = dictionaries[name].to(torch_dtype)

        print(f"\rLoaded layer {layer}", end='', flush=True)
    

    submodules = []
    outputs = {}

    with model.trace("_", use_cache=False):
        for name, module in model.named_modules():
            name = name[1:] # remove the leading .
            if (name in dictionaries.keys()):
                outputs[name] = module.output.save()

    for name, module in model.named_modules():        
        name = name[1:] # remove the leading .
        if (name in dictionaries.keys()):
            s = Submodule(
                path=name, 
                module=module, 
                is_tuple=isinstance(outputs[name].shape, tuple),
                dictionary=dictionaries[name]
        )
            submodules.append(s)

    return model, submodules