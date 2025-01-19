import os
import json
import asyncio

from torchtyping import TensorType
from neuronpedia import fetch_all_features, NeuronpediaRequest, NeuronpediaResponse
from collections import defaultdict
from typing import List, Dict, Union
from caching import cache_activations
from nnsight import Envoy
from sae_lens import SAE
import torch
import pickle

from config import config
from caching import load_activations

class NeuronDB:
    def __init__(self):
        self.connect()

        self.loaded_features = {}

    def connect(self):
        self.db_home = config['db_home']
        os.makedirs(self.db_home, exist_ok=True)
        header_path = os.path.join(self.db_home, "header.json")

        if not os.path.exists(header_path):
            self.header = {}
        else:
            self.header = json.load(open(header_path, "r"))

        self.available()

    def load(self, path: str, tokens: TensorType["batch", "seq"] = None):
        self.loaded_features = load_activations(path, tokens)

    def available(self):
        print(self.header)

    def show(self, path, index):
        try:
            from IPython.display import HTML, display
        except ImportError:
            print("IPython is required for HTML display. Please install it or run this in a Jupyter notebook.")
            return

        feature_data = self.loaded_features[path][index]
        
        html = ""
        
        display(HTML(html))

    def _save_neuronpedia(
        self,
        save_path: str,
        raw: List[NeuronpediaResponse],
    ):
        results = defaultdict(list)

        for response in raw:
            results[response.layer_id].append(response.dict())

        for layer_id, layer_data in results.items():
            with open(os.path.join(save_path, f"{layer_id}.pkl"), "wb") as f:
                pickle.dump(layer_data, f)

    def cache_neuronpedia(
        self,
        request: NeuronpediaRequest,
    ):
        raw = asyncio.run(fetch_all_features(request))

        results = defaultdict(list)

        for response in raw:
            results[response.layer_id].append(response.dict())

        for layer_id, layer_data in results.items():
            file_name = f"{layer_id}.pkl"
            save_path = os.path.join(self.db_home, file_name)
            with open(save_path, "wb") as f:
                pickle.dump(layer_data, f)
                
    def cache_torch(
        self,
        submodule_dict: Dict[Envoy, SAE],
        tokens: Union[TensorType["batch", "seq"], str],
        **kwargs,
    ):
        if isinstance(tokens, str):
            tokens = torch.load(tokens)

        cache = cache_activations(submodule_dict, tokens, **kwargs)

        cache.save_to_disk(
            self.db_home, 
            tokens_path=tokens if isinstance(tokens, str) else None
        )