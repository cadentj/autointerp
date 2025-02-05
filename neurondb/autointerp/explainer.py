import re

from transformers import AutoTokenizer

from ..schema.base import Feature, Example
from .prompts.explainer_prompt import build_prompt
from .clients import HTTPClient

def _highlight(index: int, example: Example, activation_threshold: float):
    result = f"Example {index}: "

    activations = example.activations
    str_toks = example.str_tokens

    def check(i):
        return activations[i] > activation_threshold

    i = 0

    while i < len(str_toks):
        if check(i):
            result += "<<"

            while i < len(str_toks) and check(i):
                result += str_toks[i]
                i += 1
            result += ">>"
        else:
            result += str_toks[i]
            i += 1

    return "".join(result)


def _get_toks_and_acts(example: Example, activation_threshold: float):
    mask = example.activations > activation_threshold

    normalized = example.normalized_activations[mask].tolist()

    return zip(example.str_tokens, normalized)


def _parse_explanation(text: str) -> str:
    try:
        match = re.search(r"\[EXPLANATION\]:\s*(.*)", text, re.DOTALL)
        return (
            match.group(1).strip()
            if match
            else "Explanation could not be parsed."
        )
    except Exception as e:
        print(f"Explanation parsing regex failed: {e}")
        raise


def _build_prompt(feature: Feature, activation_threshold: float, use_cot: bool):
    highlighted_examples = []

    for i, example in enumerate(feature.examples):
        index = i + 1  # Start at 1
        formatted = _highlight(index, example, activation_threshold)

        acts = ", ".join(
            f'("{item[0]}", {item[1]})'
            for item in _get_toks_and_acts(example, activation_threshold)
        )
        formatted += "\nActivations: " + acts

        highlighted_examples.append(formatted)

    highlighted_examples = "\n".join(highlighted_examples)

    return build_prompt(
        examples=highlighted_examples,
        use_cot=use_cot,
    )


def _prepare_examples(feature: Feature, tokenizer: AutoTokenizer):
    for example in feature.examples:
        example.str_tokens = tokenizer.batch_decode(example.tokens)

    return feature


async def explain(feature: Feature, threshold: float, client: HTTPClient, tokenizer: AutoTokenizer, use_cot: bool = False, **generation_kwargs):
    activation_threshold = feature.max_activation * threshold
    feature = _prepare_examples(feature, tokenizer)
    messages = _build_prompt(feature, activation_threshold, use_cot)

    response = await client.generate(
        messages, **generation_kwargs
    )

    try:
        return _parse_explanation(response)

    except Exception as e:
        print(f"Explanation parsing failed: {e}")
        return "Explanation could not be parsed."
