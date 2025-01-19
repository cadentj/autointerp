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

from circuitsvis.tokens import colored_tokens

from config import config
from caching import load_activations

class NeuronDB:
    def __init__(self):
        self.connect()

    def connect(self):
        self.db_home = config['db_home']
        os.makedirs(self.db_home, exist_ok=True)
        header_path = os.path.join(self.db_home, "header.json")

        if not os.path.exists(header_path):
            self.header = {}
        else:
            self.header = json.load(open(header_path, "r"))

        self.available()

    def available(self):
        print(json.dumps(self.header, indent=4))

    def commit(self):
        with open(os.path.join(self.db_home, "header.json"), "w") as f:
            json.dump(self.header, f)

    def show(self, save_id, index, max_examples=5, **load_kwargs):
        try:
            from IPython.display import HTML, display
        except ImportError:
            print("IPython is required for HTML display. Please install it or run this in a Jupyter notebook.")
            return

        save_path = self.header[save_id]
        is_neuronpedia = os.path.dirname(save_path).split("/")[-1] == "neuronpedia"
        # feature_data = self.loaded_features[save_id][index]

        if is_neuronpedia:
            tokens, activations = self.load_neuronpedia(save_path, index, max_examples)
        else:
            tokens, activations = self.load_torch(save_path, index, max_examples, **load_kwargs)
        
        # html = ""
        
        # display(HTML(html))

        for token, activation in zip(tokens, activations):
            display(colored_tokens(token, activation))

    def load_torch(self, path: str, index: int, max_examples, **load_kwargs):
        tokens, activations = load_activations(path, index=index, max_examples=max_examples, **load_kwargs)[index]

        return tokens, activations

    def load_neuronpedia(self, path: str, index: int, max_examples):
        with open(path, "rb") as f:
            features = pickle.load(f)

        feature = [NeuronpediaResponse(**f) for f in features if f['index'] == index][0]
        activations = [act.values for act in feature.activations[:max_examples]]
        tokens = [act.tokens for act in feature.activations[:max_examples]]

        return tokens, activations

    def cache_neuronpedia(
        self,
        request: NeuronpediaRequest,
    ):
        raw = asyncio.run(fetch_all_features(request))

        results = defaultdict(list)

        for response in raw:
            results[response.layer_id].append(response.model_dump())

        neuronpedia_dir = os.path.join(self.db_home, "neuronpedia")
        os.makedirs(neuronpedia_dir, exist_ok=True)

        for layer_id, layer_data in results.items():
            file_name = f"{layer_id}.pkl"
            save_path = os.path.join(neuronpedia_dir, file_name)

            with open(save_path, "wb") as f:
                pickle.dump(layer_data, f)

            self.header[layer_id] = save_path

        self.commit()
                
    def cache_torch(
        self,
        model,
        submodule_dict: Dict[Envoy, SAE],
        tokens: Union[TensorType["batch", "seq"], str],
        **kwargs,
    ):
        if isinstance(tokens, str):
            loaded_tokens = torch.load(tokens)

        cache = cache_activations(model, submodule_dict, loaded_tokens, **kwargs)

        torch_dir = os.path.join(self.db_home, "torch")
        os.makedirs(torch_dir, exist_ok=True)

        for module_path, save_path in cache.save_to_disk(
            torch_dir, 
            tokens_path=tokens if isinstance(tokens, str) else None
        ):
            self.header[module_path] = save_path

        self.commit()
