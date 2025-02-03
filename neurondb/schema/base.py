from typing import List, Dict, NamedTuple

from torchtyping import TensorType


DictionaryRequest = Dict[str, List[int]]

class Example(NamedTuple):
    tokens: TensorType["seq"]
    activations: TensorType["seq"]
    normalized_activations: TensorType["seq"]
    
class Feature(NamedTuple):
    index: int
    max_activation: float
    examples: List[Example]