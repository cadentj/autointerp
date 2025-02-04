from .prompts.simulation_prompt import format_prompt
from ..schema.base import Example

from transformers import AutoTokenizer

from .clients import LogProbsClient
from typing import List, Tuple


def _parse_simulation_response(
    simulation_results: List[Tuple[Example, List[float], str]], tab_id: int, tokenizer: AutoTokenizer
) -> float:
    for example, logits, prompt in simulation_results:
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens = False)
        tab_probs = [
            prob for prob, tok in zip(logits, prompt_tokens) if tok == tab_id
        ]



def simulate(
    explanation: str,
    examples: List[Example],
    tokenizer: AutoTokenizer,
    client: LogProbsClient,
) -> float:
    prompts = [
        format_prompt(explanation, example, tokenizer) for example in examples
    ]
    tab_id = tokenizer.encode("\t", add_special_tokens=False)[0]

    example_logits = [
        (example, logits, prompt)
        for example, logits, prompt in zip(examples, client.generate(prompts), prompts)
    ]

    result = _parse_simulation_response(example_logits, tab_id, tokenizer)

    return result
