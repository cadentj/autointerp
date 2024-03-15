from dataclasses import dataclass
from torch import Tensor
from typing import List
import torch

from prompts import system_prompt, few_shot_explainations

@dataclass
class Feature:

    acts: Tensor
    normalized_acts: Tensor = None
    tokens: List[str]
    explaination: str


def parse_activation(activation: Feature) -> str:
    pass

def build_explainer_prompt(
    tokenizer,
    few_shot: List[Feature],
    template: str,
):
    
    messages = [
        {
            "role" : "system",
            "content" : system_prompt
        }
    ]

    for feature in few_shot_explainations:

        messages.append(
            {
                "role" : "user",
                "content" : parse_activation(feature)
            }
        )

        messages.append(
            {
                "role" : "assistant", 
                "content" : feature.explaination
            }
        )



### EXPLAINER PROMPTS ###

system_prompt = """We're studying neurons in a neural network. Each neuron looks for some particular thing in a short document. Look at the parts of the document the neuron activates for and summarize in a single sentence what the neuron is looking for. Don't list examples of words.

The activation format is token<tab>activation. Activation values range from 0 to 10. A neuron finding what it's looking for is represented by a non-zero activation value. The higher the activation value, the stronger the match."""

few_shot_explainations = [
    Feature(
        acts=None,
        normalized_acts=torch.Tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        tokens=["The", "neuron", "is", "looking", "for", "a", "person."],
        explaination="The neuron is looking for a person."
    )
]