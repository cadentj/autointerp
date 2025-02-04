"""Original implementation by Bills et al. 2023.

https://github.com/openai/automated-interpretability/blob/main/neuron-explainer/neuron_explainer/explanations/simulator.py
"""

from collections import OrderedDict
from typing import List, Tuple

import numpy as np
from torchtyping import TensorType
from transformers import AutoTokenizer

from .clients import NsClient
from .prompts.simulation_prompt import format_prompt
from ..schema.base import Example
from ..schema.client import PromptLogProbs

VALID_ACTIVATION_TOKENS = list(str(i) for i in range(9 + 1))
ACTIVATION_TOKEN_IDS = None


def _correlation_score(
    real_activations: List[float] | np.ndarray,
    predicted_activations: List[float] | np.ndarray,
) -> float:
    return np.corrcoef(real_activations, predicted_activations)[0, 1]


def _compute_expected_value(
    norm_probabilities_by_distribution_value: OrderedDict[int, float],
) -> float:
    """
    Given a map from distribution values (integers on the range [0, 9]) to normalized
    probabilities, return an expected value for the distribution.
    """
    return np.dot(
        np.array(list(norm_probabilities_by_distribution_value.keys())),
        np.array(list(norm_probabilities_by_distribution_value.values())),
    )


def _parse_top_log_probs(
    indices: TensorType["top_k"], values: TensorType["top_k"]
):
    """For a specific token's top-k log probabilities/indices, return a dictionary
    of distribution values (integers on the range [0, 9]) to their
    raw probabilities (after applying exp to convert from log probs).
    """
    probabilities_by_distribution_value = OrderedDict()
    for idx, log_prob in zip(indices, values):
        if idx in ACTIVATION_TOKEN_IDS:
            token_str = ACTIVATION_TOKEN_IDS[idx]
            token_as_int = int(token_str)
            # Convert from log prob to prob using exp
            probabilities_by_distribution_value[token_as_int] = np.exp(
                log_prob
            )
    return probabilities_by_distribution_value


def _get_expected_value(
    indices: TensorType["top_k"], values: TensorType["top_k"]
):
    """For a specific token's top-k probabilities/indices, compute the expected
    value of the distribution.
    """

    # Get the probabilities of the distribution values
    probabilities_by_distribution_value = _parse_top_log_probs(indices, values)

    # Normalize the probabilities to sum to 1
    total_p_of_distribution_values = sum(
        probabilities_by_distribution_value.values()
    )
    norm_probabilities_by_distribution_value = OrderedDict(
        {
            distribution_value: (p / total_p_of_distribution_values)
            for distribution_value, p in probabilities_by_distribution_value.items()
        }
    )

    # Compute the expected value
    expected_value = _compute_expected_value(
        norm_probabilities_by_distribution_value
    )

    return expected_value


def _score(
    true_activations: List[float],
    simulation_result: PromptLogProbs,
    tab_id: int,
) -> float:
    """Computes correlation scores between true activations and predicted values."""
    expected_values = []

    # For each token in the simulated example, compute the expected value
    for idxs, token_log_probs, tok in zip(
        simulation_result.indices,
        simulation_result.values,
        simulation_result.tokens,
    ):
        if tok == tab_id:
            expected_value = _get_expected_value(idxs, token_log_probs)
            expected_values.append(expected_value)

    # Compute the correlation score between the true activations and the expected values
    ev_correlation_score = _correlation_score(
        true_activations, expected_values
    )

    return ev_correlation_score, expected_values


def _parse_and_score(
    true_activations: List[List[float]],
    simulation_results: List[PromptLogProbs],
    tab_id: int,
) -> Tuple[List[float], float]:
    """Computes correlation scores between true activations and predicted values.

    Takes two inputs:
    - true_activations: The actual neuron activation values
    - simulation_results: List of PromptLogProbs containing top-k token log probabilities

    For each token position:
    1. Extracts log probabilities for tokens 0-9
    2. Converts to raw probabilities and normalizes them to sum to 1
    3. Computes expected value as sum(digit * probability)

    Returns:
    - Per-example correlation scores between true and predicted activations
    - Overall correlation score across all examples
    """
    all_expected_values = []
    per_example_correlation_scores = []

    # For each scored sequence example, compute the expected value and correlation score
    for true_acts, simulation_result in zip(
        true_activations, simulation_results
    ):
        ev_correlation_score, expected_values = _score(
            true_acts, simulation_result, tab_id
        )
        per_example_correlation_scores.append(ev_correlation_score)
        all_expected_values.extend(expected_values)

    # Compute the correlation score across all examples
    all_ev_correlation_score = _correlation_score(
        np.array(true_activations).flatten(),
        np.array(all_expected_values),
    )

    return per_example_correlation_scores, all_ev_correlation_score


def _setup_activation_token_ids(tokenizer: AutoTokenizer):
    global ACTIVATION_TOKEN_IDS
    ACTIVATION_TOKEN_IDS = {
        tokenizer.encode(token, add_special_tokens=False)[0]: token
        for token in VALID_ACTIVATION_TOKENS
    }


def simulate(
    explanation: str,
    examples: List[Example],
    client: NsClient,
    subject_tokenizer: AutoTokenizer,
) -> float:
    simulator_tokenizer = client.tokenizer

    _setup_activation_token_ids(simulator_tokenizer)
    tab_id = simulator_tokenizer.encode("\t", add_special_tokens=False)[0]

    prompts = [
        format_prompt(explanation, example, subject_tokenizer)
        for example in examples
    ]

    responses = [
        prompt_log_probs for prompt_log_probs in client.generate(prompts)
    ]

    true_activations = [example.activations.tolist() for example in examples]
    result = _parse_and_score(true_activations, responses, tab_id)

    return result
