import re
from typing import List

from ..schema.base import Feature, Example
from .prompts.explainer_prompt import build_prompt


class Query:
    def __init__(
        self,
        client,
        tokenizer,
        use_cot: bool = False,
        threshold: float = 0.6,
        **generation_kwargs,
    ):
        self.client = client
        self.tokenizer = tokenizer

        self.use_cot = use_cot
        self.threshold = threshold
        self.generation_kwargs = generation_kwargs

    async def __call__(self, query: str, feature: Feature):
        self.activation_threshold = feature.max_activation * self.threshold
        messages = self._build_prompt(query, feature)


        with open("prompt.txt", "w") as f:
            f.write(messages[-1]["content"])

        response = await self.client.generate(
            messages, **self.generation_kwargs
        )

        try:
            return self._parse_score(response)

        except Exception as e:
            print(f"Explanation parsing failed: {e}")
            return "Explanation could not be parsed."

    def _get_toks_and_acts(self, example: Example, max_activation: float):
        mask = example.activations > self.activation_threshold

        activations = example.activations[mask]
        normalized = (activations / max_activation) * 10
        normalized = normalized.round().int().tolist()

        tokens = example.tokens[mask]
        str_toks = self.tokenizer.batch_decode(tokens)

        return zip(str_toks, normalized)

    def _build_prompt(self, query: str, feature: Feature):
        highlighted_examples = []

        for i, example in enumerate(feature.examples):
            index = i + 1  # Start at 1
            formatted = self._highlight(index, example)

            acts = ", ".join(
                f'("{item[0]}", {item[1]})'
                for item in self._get_toks_and_acts(
                    example, feature.max_activation
                )
            )
            formatted += "\nActivations: " + acts

            highlighted_examples.append(formatted)

        highlighted_examples = "\n".join(highlighted_examples)

        return build_prompt(
            examples=highlighted_examples,
            query=query,
            use_cot=self.use_cot,
        )

    def _parse_score(self, text: str) -> str:
        try:
            match = re.search(r"\[SCORE\]:\s*(.*)", text, re.DOTALL)
            return (
                match.group(1).strip()
                if match
                else "Score could not be parsed."
            )
        except Exception as e:
            print(f"Score parsing regex failed: {e}")
            raise

    def _highlight(self, index, example):
        result = f"Example {index}: "

        activations = example.activations
        str_toks = self.tokenizer.batch_decode(example.tokens)

        def check(i):
            return activations[i] > self.activation_threshold

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
