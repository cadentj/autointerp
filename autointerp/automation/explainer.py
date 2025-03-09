import re
from collections import defaultdict

from .prompts.explainer_prompt import build_prompt
from .clients import HTTPClient
from ..base import Feature


class Explainer:
    def __init__(
        self,
        client: HTTPClient,
        insert_as_prompt: bool = False,
        verbose: bool = False,
    ):
        self.client = client
        self.verbose = verbose
        self.insert_as_prompt = insert_as_prompt

    async def __call__(self, feature: Feature, **generation_kwargs):
        messages = self._build_prompt(feature)

        response = await self.client.generate(messages, **generation_kwargs)

        if self.verbose:
            with open(f"response-{feature.index}.txt", "w") as f:
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
        high_activating_examples = []
        middle_activating_examples = []
        low_activating_examples = []

        sorted_examples = defaultdict(list)
        for example in feature.activating_examples:
            sorted_examples[example.quantile].append(example)

        for quantile, examples in sorted_examples.items():
            index = 0
            for example in examples:
                index += 1
                formatted = self._highlight(index, example)
                if quantile == 3:
                    high_activating_examples.append(formatted)
                elif quantile == 2:
                    middle_activating_examples.append(formatted)
                elif quantile == 1:
                    low_activating_examples.append(formatted)
                else:
                    raise ValueError(f"Invalid quantile: {quantile}")

        return build_prompt(
            high_activating_examples="\n".join(high_activating_examples),
            middle_activating_examples="\n".join(middle_activating_examples),
            low_activating_examples="\n".join(low_activating_examples),
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

        def check(i):
            return activations[i] > 0.0

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
