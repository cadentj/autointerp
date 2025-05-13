import re
from typing import List, Literal

from .prompts.explainer_prompt import build_prompt
from .clients import HTTPClient
from ..base import Feature


class Explainer:
    def __init__(
        self,
        client: HTTPClient,
        max_or_min: Literal["max", "min"] = "max",
        threshold: float = 0.0,
        insert_as_prompt: bool = False,
        verbose: bool = False,
    ):
        self.client = client
        self.verbose = verbose
        self.insert_as_prompt = insert_as_prompt

        self.max_or_min = max_or_min
        self.threshold = threshold

    async def __call__(
        self,
        feature: Feature,
        **generation_kwargs,
    ):
        messages = self._build_prompt(feature)

        response = await self.client.generate(messages, **generation_kwargs)

        if self.verbose:
            with open(f"response-{feature.index}-{self.max_or_min}.txt", "w") as f:
                for message in messages:
                    f.write(f"{message['role'].upper()}:\n\n")
                    f.write(message["content"] + "\n\n")
                f.write(response)

        try:
            return self._parse_explanation(response)

        except Exception as e:
            print(f"Explanation parsing failed: {e}")
            return "Explanation could not be parsed."

    def _build_prompt(self, feature: Feature):
        if self.max_or_min == "max":
            examples = feature.max_activating_examples
        else:
            examples = feature.min_activating_examples

        formatted_examples = [
            self._highlight(index + 1, example)
            for index, example in enumerate(examples)
        ]

        return build_prompt(
            examples="\n".join(formatted_examples),
            insert_as_prompt=self.insert_as_prompt,
        )

    def _parse_explanation(self, text: str) -> str:
        try:
            match = re.search(r"\\boxed\{(.*?)\}", text)
            return (
                match.group(1).strip()
                if match
                else "Explanation could not be parsed."
            )
        except Exception as e:
            print(f"Explanation parsing regex failed: {e}")
            raise

    def _highlight(self, index, example):
        result = f"Example {index}: "

        activations = example.activations
        str_toks = example.str_tokens

        abs_activations = activations.abs()
        max_index = abs_activations.argmax().item()
        max_sign = activations[max_index] > 0

        if max_sign:
            should_highlight = lambda i: activations[i] > activations[max_index] * self.threshold
        else:
            should_highlight = lambda i: activations[i] < activations[max_index] * self.threshold
        
        i = 0

        while i < len(str_toks):
            if should_highlight(i):
                result += "<<"

                while i < len(str_toks) and should_highlight(i):
                    result += str_toks[i]
                    i += 1
                result += ">>"
            else:
                result += str_toks[i]
                i += 1

        return "".join(result)
