from nnsight import Envoy
from typing import NamedTuple, Union, List
from sae_lens import SAE
import json
import torch as t
from .sae import GlobalBatchTopKMatryoshkaSAE
import os


class MatryoshkaSAEConfig(NamedTuple):
    device: str
    model_dtype: t.dtype
    model_name: str
    hook_point: str
    layer: int
    d_sae: int
    top_k: int
    top_k_aux: int
    aux_penalty: float
    bandwidth: float
    group_sizes: List[int]
    top_k_matryoshka: List[int]
    dict_size: int
    group_sizes: List[int]
    input_unit_norm: bool
    seed: int
    batch_size: int
    lr: float
    num_tokens: float
    l1_coeff: float
    beta1: float
    beta2: float
    max_grad_norm: float
    seq_len: int
    dtype: t.dtype
    model_dtype: t.dtype
    model_batch_size: int
    num_batches_in_buffer: int
    dataset_path: str
    wandb_project: str
    perf_log_freq: int
    checkpoint_freq: int
    n_batches_to_dead: int
    name: str

class MatryoshkaSAE(GlobalBatchTopKMatryoshkaSAE):
    def __init__(self, cfg):
        super().__init__(cfg)
        
        if "resid_pre" in cfg["hook_point"]:
            cfg["layer"] = cfg["layer"] - 1
        cfg["d_sae"] = cfg["dict_size"]
        
        # Filter cfg to only include fields needed for MatryoshkaSAEConfig
        config_fields = {field: cfg[field] for field in MatryoshkaSAEConfig._fields}
        self.cfg = MatryoshkaSAEConfig(**config_fields)

class Submodule(NamedTuple):
    path: str
    module: Envoy
    is_tuple: bool
    dictionary: Union[SAE, MatryoshkaSAE]

def load_matryoshka_sae(layer, device):

    config_path = f"/share/u/caden/neurondb/neurondb/matryoshka/config.json"
    with open(config_path, "r") as f:
        cfg = json.load(f)

    hook_point = cfg["hook_point"]
    hook_layer = int(''.join(c for c in hook_point if c.isdigit()))
    if 'resid_pre' in hook_point:
        hook_layer -= 1
    if layer != hook_layer:
        raise ValueError(f"Layer {layer} does not match hook point {hook_point}")

    # Convert string representations back to torch.dtype
    if "dtype" in cfg:
        cfg["dtype"] = getattr(t, cfg["dtype"].split(".")[-1])

    # Convert group_sizes back to a list if it's a string
    if isinstance(cfg['group_sizes'], str):
        cfg['group_sizes'] = json.loads(cfg['group_sizes'])

    if isinstance(cfg["top_k_matryoshka"], str):
        cfg["top_k_matryoshka"] = json.loads(cfg["top_k_matryoshka"])

    sae = MatryoshkaSAE(cfg)
    sae.to(device)
    return sae