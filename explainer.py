from typing import List
from torch import Tensor

class Explainer:
    def __init__(self, model, data, target):
        self.model = model
        self.data = data
        self.target = target

    def explain(self, instance):
        pass


    def _normalize_activations(
        self,
        tokens: List[str],
        activations: Tensor,
    ):
        # Normalize activations to integers between 0 and 10
        activations = activations - activations.min()
        activations = activations / activations.max()
        activations = (activations * 10).int()
        
