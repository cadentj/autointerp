import asyncio
import json
import random
import re
from typing import (
    Any,
    Callable,
    List,
    Literal,
    NamedTuple,
    Sequence,
)

import torch
from collections import defaultdict
from ..base import Feature, Example
from .clients import HTTPClient, Response
from .prompts.detection_prompt import prompt as detection_prompt
from .prompts.fuzz_prompt import prompt as fuzz_prompt

L = "<<"
R = ">>"
DEFAULT_MESSAGE = (
    "<<NNsight>> is the best library for <<interpretability>> on huge models!"
)


class Sample(NamedTuple):
    text: str
    activating: bool
    quantile: int


def examples_to_samples(
    examples: List[Example],
    n_incorrect: int = 0,
    threshold: float = 0.3,
    highlighted: bool = False,
) -> List[Sample]:
    samples = []

    for example in examples:
        str_toks = example.str_tokens
        text = _prepare_text(
            example, str_toks, n_incorrect, threshold, highlighted
        )

        # IMPORTANT NOTE:
        # Activating means whether the example's ground truth is to be "correct" or "incorrect"
        # The first condition is for fuzzing. If the example has incorrectly marked tokens, it is not activating.
        # Also note that fuzzed examples should never have a quantile of -1 because the prompt entailment is that
        # the example is activating.
        # The second condition is for detection. A quantile of -1 means that the example is not activating.
        activating = (highlighted and n_incorrect == 0) or (
            not highlighted and example.quantile != -1
        )

        samples.append(
            Sample(
                text=text,
                activating=activating,
                quantile=example.quantile,
            )
        )

    return samples


def _prepare_text(
    example: Example,
    str_toks: List[str],
    n_incorrect: int,
    threshold: float,
    highlighted: bool,
) -> str:
    # 1) Return full text for detection
    if not highlighted:
        clean = "".join(str_toks)
        return clean

    threshold = threshold * example.activations.max()

    # 2) Highlight tokens with activations above threshold if correct example
    if n_incorrect == 0:

        def threshold_check(i):
            return example.activations[i] >= threshold

        return _highlight(str_toks, threshold_check)

    below_threshold = torch.nonzero(example.activations <= threshold).squeeze(
        -1
    )

    # Add check for empty tensor
    if below_threshold.numel() == 0:
        print("Failed to prepare example - no tokens below threshold.")
        return DEFAULT_MESSAGE

    # 4) Highlight n_incorrect tokens with activations below threshold
    n_incorrect = min(n_incorrect, below_threshold.numel())

    random_indices = set(random.sample(below_threshold.tolist(), n_incorrect))
    return _highlight(str_toks, lambda i: i in random_indices)


def _highlight(tokens: Sequence[str], check: Callable[[int], bool]) -> str:
    result = []

    i = 0
    while i < len(tokens):
        if check(i):
            result.append(L)

            while i < len(tokens) and check(i):
                result.append(tokens[i])
                i += 1

            result.append(R)
        else:
            result.append(tokens[i])
            i += 1

    return "".join(result)


class Classifier:
    def __init__(
        self,
        client: HTTPClient,
        n_examples_shown: int = 10,
        method: Literal["detection", "fuzzing"] = "detection",
        threshold: float = 0.3,
        verbose: bool = False,
    ):
        """Initialize a Classifier.

        Args:
            client: The client to use for generation
            n_examples_shown: Number of examples to show in prompt
            log_prob: Whether to use log probabilities for AUC calculation
            method: Classification method - "detection" or "fuzzing"
            threshold: Activation threshold for fuzzing
            temperature: Temperature for generation
        """
        self.client = client
        self.n_examples_shown = n_examples_shown
        self.method = method
        self.threshold = threshold
        self.verbose = verbose

    async def __call__(
        self,
        feature: Feature,
        explanation: str,
        **generation_kwargs: Any,
    ):
        """Run classification on a feature"""
        prepare = getattr(self, f"_prepare_{self.method}")
        samples = prepare(feature)
        random.shuffle(samples)

        batches = [
            samples[i : i + self.n_examples_shown]
            for i in range(0, len(samples), self.n_examples_shown)
        ]
        tasks = [
            self._generate(explanation, batch, **generation_kwargs)
            for batch in batches
        ]
        results = await asyncio.gather(*tasks)
        return self._grade(results, batches)

    def _grade(
        self, results: List[List[bool]], batches: List[List[Sample]]
    ) -> dict:
        per_quantile_results = defaultdict(
            lambda: {"TP": 0, "FN": 0, "FP": 0, "TN": 0}
        )
        for batch, results in zip(batches, results):
            for sample, prediction in zip(batch, results):
                match (sample.activating, prediction == 1):
                    case (True, True):
                        per_quantile_results[sample.quantile]["TP"] += 1
                    case (True, False):
                        per_quantile_results[sample.quantile]["FN"] += 1
                    case (False, True):
                        per_quantile_results[sample.quantile]["FP"] += 1
                    case (False, False):
                        per_quantile_results[sample.quantile]["TN"] += 1

        return dict(per_quantile_results)

    def _prepare_detection(self, feature: Feature) -> List[Sample]:
        """Prepare samples for detection method"""
        random_samples = examples_to_samples(
            feature.non_activating_test_examples,
        )

        activating_samples = examples_to_samples(
            feature.activating_test_examples,
        )

        return random_samples + activating_samples

    def _prepare_fuzzing(self, feature: Feature) -> List[Sample]:
        """Prepare samples for fuzzing method"""
        examples = feature.examples
        random.shuffle(examples)

        # Calculate the mean number of activations in the examples and convert to int
        n_incorrect = int(
            sum(
                len(torch.nonzero(example.activations)) for example in examples
            )
            / len(examples)
        )

        quantiles = set(example.quantile for example in examples)
        binned = {quantile: [] for quantile in quantiles}
        for example in examples:
            binned[example.quantile].append(example)

        n_activating = len(binned[1]) // 2
        activating_examples = [
            example
            for quantile in binned.values()
            for example in quantile[:n_activating]
        ]
        non_activating_examples = [
            example
            for quantile in binned.values()
            for example in quantile[n_activating:]
        ]

        random_samples = examples_to_samples(
            non_activating_examples,
            n_incorrect=n_incorrect,
            highlighted=True,
            threshold=self.threshold,
        )

        activating_samples = examples_to_samples(
            activating_examples,
            n_incorrect=0,
            highlighted=True,
            threshold=self.threshold,
        )

        return random_samples + activating_samples

    async def _generate(
        self, explanation: str, batch: List[Sample], **generation_kwargs: Any
    ):
        """
        Generate predictions for a batch of samples.
        """

        examples = "\n".join(
            f"Example {i + 1}: {sample.text}" for i, sample in enumerate(batch)
        )

        prompt_template = (
            detection_prompt if self.method == "detection" else fuzz_prompt
        )

        prompt = prompt_template(examples, explanation)

        response = await self.client.generate(prompt, **generation_kwargs)

        if self.verbose:
            with open("response.txt", "a") as f:
                f.write(f"PROMPT\n{prompt}\n\n")
                f.write(f"RESPONSE\n{response}\n\n")
                f.write(f"TRUE\n{[i.activating for i in batch]}\n\n")

        return self._parse(response)

    def _parse(self, response: Response) -> List[bool]:
        pattern = r"\[.*?\]"
        match = re.search(pattern, response)

        if match is None:
            print(f"No match found in response: {response}")
            return [0] * self.n_examples_shown

        try:
            result = json.loads(match.group(0))

            if len(result) != self.n_examples_shown:
                print(f"Incorrect number of results: {len(result)}")
                return [0] * self.n_examples_shown
            return result

        except json.JSONDecodeError:
            print(f"JSONDecodeError: {response}")
            return [0] * self.n_examples_shown
