from typing import List, Dict, NamedTuple

from torchtyping import TensorType
from transformers import AutoTokenizer

DictionaryRequest = Dict[str, List[int]]

class Example(NamedTuple):
    tokens: TensorType["seq"]
    activations: TensorType["seq"]
    normalized_activations: TensorType["seq"]

    
class Feature(NamedTuple):
    index: int
    max_activation: float
    examples: List[Example]

    def display(
        self,
        tokenizer: AutoTokenizer,
        threshold: float = 0.0,
        n: int = 10,
    ) -> str:
        from IPython.core.display import HTML, display

        def _to_string(tokens: TensorType["seq"], activations: TensorType["seq"]) -> str:
            result = []
            i = 0

            max_act = activations.max()
            _threshold = max_act * threshold

            while i < len(tokens):
                if activations[i] > _threshold:
                    result.append("<mark>")
                    while i < len(tokens) and activations[i] > _threshold:
                        result.append(tokens[i])
                        i += 1
                    result.append("</mark>")
                else:
                    result.append(tokens[i])
                    i += 1
            
            return "".join(result)
        
        strings = [
            _to_string(tokenizer.batch_decode(example.tokens), example.activations)
            for example in self.examples[:n]
        ]

        display(HTML("<br><br>".join(strings)))
