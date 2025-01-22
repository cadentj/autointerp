import os
import json
import asyncio

from torchtyping import TensorType
from collections import defaultdict
from typing import List, Dict, Union, Tuple

from nnsight import Envoy
from sae_lens import SAE
import torch
import pickle

from .config import config
from .doll_caching import load_activations, cache_activations
from .neuronpedia import fetch_all_features, NeuronpediaRequest, NeuronpediaResponse

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

    def available(self) -> None:
        print(json.dumps(self.header, indent=4))

    def commit(self) -> None:
        with open(os.path.join(self.db_home, "header.json"), "w") as f:
            json.dump(self.header, f)

    def show(self, save_id, index, max_examples=5, **load_kwargs) -> None:
        save_path = self.header[save_id]
        is_neuronpedia = os.path.dirname(save_path).split("/")[-1] == "neuronpedia"

        if is_neuronpedia:
            tokens, activations = self.load_neuronpedia(save_path, index, max_examples)
        else:
            tokens, activations = self.load_torch(save_path, index, max_examples, **load_kwargs)
        
        from .vis import show_neuron
        show_neuron(tokens, activations, max_examples)

    def export_torch(self, request: Dict[str, List[int]], output_path: str = "vis.html", max_examples=5, show=False, **load_kwargs) -> None:
        """Export neuron activations to an HTML file with highlighted tokens.
        
        Args:
            request: Dictionary mapping layer paths to lists of feature indices
            output_path: Path to save the HTML file
            max_examples: Maximum number of examples to show per neuron
            **load_kwargs: Additional arguments passed to load_activations
        """
        neurons_data = []
        
        for layer_id, indices in request.items():
            if isinstance(indices, torch.Tensor):
                indices = indices.tolist()
            
            for index in indices:
                # Skip max examples here to load all, then split
                tokens, activations, max_activation, _ = self.load_torch(self.header[layer_id], index, **load_kwargs)
                tokens, activations = self._split_activations(tokens, activations, max_examples)
                neurons_data.append((layer_id, index, tokens, activations, max_activation, None))

        from .vis import export_neurons, display_neurons
        html = export_neurons(neurons_data, output_path)
        if show:
            display_neurons(html)

    def export_neuronpedia(self, request: NeuronpediaRequest, output_path: str = "vis.html", max_examples=5, show=False, **load_kwargs) -> None:
        """Export neuron activations to an HTML file with highlighted tokens."""

        request = NeuronpediaRequest(**request)
        neurons_data = []
        for layer_id, indices in [(req.layer_id, req.indices) for req in request.dictionaries]:
            for index in indices:
                # Skip max examples here to load all, then split.
                tokens, activations, max_activation, pos_str = self.load_neuronpedia(self.header[layer_id], index, **load_kwargs)
                tokens, activations = self._split_activations(tokens, activations, max_examples)
                neurons_data.append((layer_id, index, tokens, activations, max_activation, pos_str))

        from .vis import export_neurons, display_neurons
        html = export_neurons(neurons_data, output_path)
        if show:
            display_neurons(html)

    def _split_activations(self, tokens, activations, max_examples):
        """Split tokens and activations into top and middle sections.
        Takes first max_examples from first half and first max_examples from second half."""
        
        half_idx = len(activations) // 2

        return (
            (tokens[:max_examples], tokens[half_idx:half_idx + max_examples]),  # First n from each half
            (activations[:max_examples], activations[half_idx:half_idx + max_examples])  # First n from each half
        )

    def load_torch(
        self, 
        path: str, 
        index: int, 
        max_examples: int = -1, 
        **load_kwargs
    ) -> Tuple[Tuple[List[str], List[str]], Tuple[TensorType["max_examples", "seq"], TensorType["max_examples", "seq"]], float, List[str]]:
        tokens, activations = load_activations(path, index=index, max_examples=max_examples, **load_kwargs)[index]

        max_activation = torch.max(activations).item()
        return tokens, activations, max_activation, None

    def load_neuronpedia(
        self, 
        path: str, 
        index: int, 
        max_examples: int = -1, 
        **load_kwargs
    ) -> Tuple[Tuple[List[str], List[str]], Tuple[TensorType["max_examples", "seq"], TensorType["max_examples", "seq"]], float, List[str]]:
        with open(path, "rb") as f:
            features = pickle.load(f)

        feature = [NeuronpediaResponse(**f) for f in features if f['index'] == index][0]
        activations = [act.values for act in feature.activations[:max_examples]]
        tokens = [act.tokens for act in feature.activations[:max_examples]]

        return tokens, activations, feature.max_activation, feature.pos_str

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
