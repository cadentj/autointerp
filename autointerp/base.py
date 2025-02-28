from dataclasses import dataclass, field
from typing import List, NamedTuple

from torchtyping import TensorType
from transformers import AutoTokenizer

class Example(NamedTuple):
    tokens: TensorType["seq"]
    """Token ids tensor."""

    str_tokens: List[str]
    """Decoded stringified tokens."""

    activations: TensorType["seq"]
    """Raw activations."""

    normalized_activations: TensorType["seq"]
    """Normalized activations. Used for similarity search."""

    quantile: int
    """Quantile of the activation. -1 if not applicable. 0 if non-activating."""
    
@dataclass
class Feature:
    index: int
    """Index of the feature in the SAE."""

    max_activation: float
    """Maximum activation of the feature across all examples."""

    activating_examples: List[Example]
    """Activating examples."""

    non_activating_examples: List[Example] = field(default_factory=list)
    """Non-activating examples."""

    @property
    def examples(self) -> List[Example]:
        return self.activating_examples + self.non_activating_examples

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
