from typing import List

import torch

from .utils import normalize_acts, Location 

class Evaluator:

    def __init__(
            self, 
            model, 
            dictionaries: List[torch.nn.Module], 
            mem: List
        ):
        self.model = model
        self.dictionaries = dictionaries

        self.mem = mem

    def __call__(
            self, 
            actions: List[str], 
            location: Location
        ) -> None:
        """Run the evaluator on some feature activating phrases for a given neuron.

        Args:
            actions (List[str]): List of phrases
            location (Location): Location of neuron to evaluate
        
        Returns:
            None
        """
        
        for a in actions:
            acts = self.get_activation(
                a, 
                location.layer, 
                location.index
            )
            score = torch.argmax(acts)

            self.mem[-1].evaluator[a] = score
    
    def get_activation(
            self, 
            action: str, 
            layer: int, 
            index: int,
        ) -> torch.Tensor: 
        """Get the normalized activations for a given phrase and neuron.

        Args:
            action (str): Phrase to evaluate
            layer (int): Layer index
            index (int): Dictionary index
        
        Returns:
            torch.Tensor: Normalized activations
        """

        with self.model.trace(action):
            activations = self.model.transformer.h[layer].input[0][0]

            _, feature_acts, _, _, _, _ = self.dictionaries[layer](activations)
            
            acts = feature_acts[:,:,index][0].save()

        acts = acts.value

        # Have to set the first act to zero bc I dont have a full context.
        acts[0] = 0.
        
        torch.cuda.empty_cache()
        return normalize_acts(acts)