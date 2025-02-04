from .prompts.simulation_prompt import format_prompt
from ..schema.base import Example

from transformers import AutoTokenizer

from .clients import NsClient
from ..schema.client import PromptProbs
from typing import List, Tuple
import numpy as np
from collections import OrderedDict
from torchtyping import TensorType

VALID_ACTIVATION_TOKENS = list(str(i) for i in range(10 + 1))
ACTIVATION_TOKEN_IDS = None


def _parse_top_probs(indices: TensorType["top_k"], values: TensorType["top_k"]):
    probabilities_by_distribution_value = OrderedDict()
    for idx, value in zip(indices, values):
        if idx in ACTIVATION_TOKEN_IDS:
            token_str = ACTIVATION_TOKEN_IDS[idx]
            token_as_int = int(token_str)
            probabilities_by_distribution_value[token_as_int] = value
    return probabilities_by_distribution_value


def _compute_expected_value(
    norm_probabilities_by_distribution_value: OrderedDict[int, float]
) -> float:
    """
    Given a map from distribution values (integers on the range [0, 10]) to normalized
    probabilities, return an expected value for the distribution.
    """
    return np.dot(
        np.array(list(norm_probabilities_by_distribution_value.keys())),
        np.array(list(norm_probabilities_by_distribution_value.values())),
    )

def _token_activation_stats(indices: TensorType["top_k"], values: TensorType["top_k"]):
    probabilities_by_distribution_value = _parse_top_probs(indices, values)
    total_p_of_distribution_values = sum(probabilities_by_distribution_value.values())
    norm_probabilities_by_distribution_value = OrderedDict(
        {
            distribution_value : p / total_p_of_distribution_values
            for distribution_value, p in probabilities_by_distribution_value.items()
        }
    )
    expected_value = _compute_expected_value(norm_probabilities_by_distribution_value)

    return (
        norm_probabilities_by_distribution_value,
        expected_value,
    )

def _parse_simulation_response(
    simulation_results: List[PromptProbs],
    tab_id: int,
) -> float:
    distribution_values = []
    distribution_probabilities = []
    expected_values = []

    for indices, values, tokens in simulation_results:
        for idxs, probs, tok in zip(indices, values, tokens):
            if tok == tab_id:
                norm_probs, expected_value = _token_activation_stats(idxs, probs)
                distribution_values.append(list(norm_probs.keys()))
                distribution_probabilities.append(list(norm_probs.values()))
                expected_values.append(expected_value)

    return (
        distribution_values,
        distribution_probabilities,
        expected_values,
    )

def _setup_activation_token_ids(tokenizer: AutoTokenizer):
    global ACTIVATION_TOKEN_IDS
    ACTIVATION_TOKEN_IDS = {
        tokenizer.encode(token, add_special_tokens=False)[0] : token
        for token in VALID_ACTIVATION_TOKENS
    }

    print("ACTIVATION_TOKEN_IDS", ACTIVATION_TOKEN_IDS)


def simulate(
    explanation: str,
    examples: List[Example],
    client: NsClient,
    subject_tokenizer: AutoTokenizer
) -> float:
    simulator_tokenizer = client.tokenizer

    _setup_activation_token_ids(simulator_tokenizer)
    tab_id = simulator_tokenizer.encode("\t", add_special_tokens=False)[0]

    prompts = [
        format_prompt(explanation, example, subject_tokenizer) for example in examples
    ]

    responses = [prompt_probs for prompt_probs in client.generate(prompts)]

    # result = _parse_simulation_response(responses, tab_id)

    # return result

    return responses
