import asyncio
import json
import random
import re
import numpy as np
from transformers import AutoTokenizer
from typing import (
    Any,
    Callable,
    List,
    Literal,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
)

import torch
from ..schema.base import Feature, Example
from ..schema.client import Response
from .clients import HTTPClient
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
    tokenizer: AutoTokenizer,
    n_incorrect: int = 0,
    threshold: float = 0.3,
    highlighted: bool = False,
) -> List[Sample]:
    samples = []

    for example in examples:
        str_toks = tokenizer.batch_decode(
            example.tokens, skip_special_tokens=True
        )
        text = _prepare_text(
            example, str_toks, n_incorrect, threshold, highlighted
        )
        activating = example.quantile != -1

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

    below_threshold = torch.nonzero(example.activations <= threshold).squeeze()

    # 3) Rare case where there are no tokens below threshold
    if below_threshold.dim() == 0:
        print("Failed to prepare example.")
        return DEFAULT_MESSAGE

    # 4) Highlight n_incorrect tokens with activations below threshold
    n_incorrect = min(n_incorrect, len(below_threshold))
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
        tokenizer: AutoTokenizer,
        n_examples_shown: int = 10,
        method: Literal["detection", "fuzzing"] = "detection",
        threshold: float = 0.3,
        temperature: float = 0.0,
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
        self.temperature = temperature
        self.tokenizer = tokenizer
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
        return np.mean(results)

    def _prepare_detection(self, feature: Feature) -> List[Sample]:
        """Prepare samples for detection method"""
        random_samples = examples_to_samples(
            feature.random_examples,
        )

        activating_samples = examples_to_samples(
            feature.examples,
        )

        return random_samples + activating_samples

    def _prepare_fuzzing(self, feature: Feature) -> List[Sample]:
        """Prepare samples for fuzzing method"""
        examples = feature.examples

        # Calculate the mean number of activations in the examples
        n_incorrect = sum(
            len(torch.nonzero(example.activations)) for example in examples
        ) / len(examples)

        random_samples = examples_to_samples(
            feature.random_examples,
            n_incorrect=n_incorrect,
            highlighted=True,
            threshold=self.threshold,
        )

        activating_samples = examples_to_samples(
            feature.examples,
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
            f"Example {i}: {sample.text}" for i, sample in enumerate(batch)
        )

        prompt_template = (
            detection_prompt if self.method == "detection" else fuzz_prompt
        )

        prompt = prompt_template(examples, explanation)

        response = await self.client.generate(
            prompt, raw=True, **generation_kwargs
        )

        if self.verbose:
            with open("response.txt", "w") as f:
                f.write(f"PROMPT\n{prompt}\n\n")
                f.write(f"RESPONSE\n{response.text}\n\n")

        predictions = self._parse(response)

        n_correct = sum(
            1
            for sample, prediction in zip(batch, predictions)
            if sample.activating == (prediction == 1)
        )

        return n_correct / len(batch)

    def _parse(
        self, response: Response
    ) -> Tuple[List[bool], List[Optional[float]]]:
        pattern = r"\[.*?\]"
        match = re.search(pattern, response.text)

        if match is None:
            return None

        return json.loads(match.group(0))
