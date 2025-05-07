from dataclasses import dataclass, field
from typing import List, NamedTuple
from enum import Enum

from torchtyping import TensorType


class NonActivatingType(Enum):
    RANDOM = "random"
    SIMILAR = "similar"


class Example(NamedTuple):
    tokens: TensorType["seq"]
    """Token ids tensor."""

    str_tokens: List[str]
    """Decoded stringified tokens."""

    activations: TensorType["seq"]
    """Raw activations."""

    normalized_activations: TensorType["seq"]
    """Normalized activations. Used for similarity search."""

    quantile: int | NonActivatingType
    """Quantile of the activation."""

def to_html(
    str_tokens: List[str],
    activations: TensorType["seq"],
    threshold: float = 0.0,
) -> str:
    result = []
    max_act = activations.max()
    _threshold = max_act * threshold

    for i in range(len(str_tokens)):
        if activations[i] > _threshold:
            # Calculate opacity based on activation value (normalized between 0.2 and 1.0)
            opacity = 0.2 + 0.8 * (activations[i] / max_act)
            result.append(
                f'<mark style="opacity: {opacity:.2f}">{str_tokens[i]}</mark>'
            )
        else:
            result.append(str_tokens[i])

    return "".join(result)

@dataclass
class Feature:
    index: int
    """Index of the feature in the SAE."""

    max_activation: float
    """Maximum activation of the feature across all examples."""

    max_activating_examples: List[Example]
    """Activating examples."""

    min_activating_examples: List[Example] = field(default_factory=list)
    """Activating examples."""

    non_activating_examples: List[Example] = field(default_factory=list)
    """Non-activating examples."""

    @property
    def examples(self) -> List[Example]:
        return self.activating_examples + self.non_activating_examples

    def display(
        self,
        threshold: float = 0.0,
        n: int = 10,
    ) -> str:
        from IPython.display import HTML, display

        strings = [
            to_html(example.str_tokens, example.activations, threshold)
            for example in self.examples[:n]
        ]

        display(HTML("<br><br>".join(strings)))

    # Alias for display method
    show = display
