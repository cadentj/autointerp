from typing import List
from torch import Tensor
import torch

from prompt_builder import Activation

class Explainer:
    def __init__(self, model, ae_list):
        self.model = model
        self.ae_list = ae_list

    def __call__(self, activation: Activation):
        
        activation.normalized_acts = self._normalize_activations(
            activation.tokens,
            activation.acts,
        )
        

    def _normalize_activations(
        self,
        tokens: List[str],
        activations: Tensor,
    ):
        # Normalize activations to integers between 0 and 10
        activations = activations - activations.min()
        activations = activations / activations.max()
        activations = (activations * 10).int()

        return activations
        

class Simulator:

    def __init__(self, model, max_trials=5):
        self.model = model

        self.reflections = []
        self.scores = []
        self.max_trials = max_trials

    def __call__(self, activation: Activation):
        # Simulate the model's response to the activation
        
        for i in range(self.max_trials):
            pass
            
                