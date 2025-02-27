from dataclasses import dataclass
from typing import List, Dict, NamedTuple, Optional

from torchtyping import TensorType
from transformers import AutoTokenizer

DictionaryRequest = Dict[str, List[int]]

class Example(NamedTuple):
    tokens: TensorType["seq"]
    str_tokens: List[str]
    activations: TensorType["seq"]
    normalized_activations: TensorType["seq"]
    quantile: Optional[int] = None
    
@dataclass
class Feature:
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
            max_act = activations.max()
            _threshold = max_act * threshold

            for i in range(len(tokens)):
                if activations[i] > _threshold:
                    # Calculate opacity based on activation value (normalized between 0.2 and 1.0)
                    opacity = 0.2 + 0.8 * (activations[i] / max_act)
                    result.append(f'<mark style="opacity: {opacity:.2f}">{tokens[i]}</mark>')
                else:
                    result.append(tokens[i])
            
            return "".join(result)
        
        strings = [
            _to_string(tokenizer.batch_decode(example.tokens), example.activations)
            for example in self.examples[:n]
        ]

        display(HTML("<br><br>".join(strings)))
