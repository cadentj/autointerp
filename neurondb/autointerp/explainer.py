import re
from typing import List

from ..schema.base import Feature, Example
from .prompts.explainer_prompt import build_prompt


class Explainer:
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

    async def __call__(self, feature: Feature):
        self.activation_threshold = feature.max_activation * self.threshold
        messages = self._build_prompt(feature.examples)

        response = await self.client.generate(
            messages, **self.generation_kwargs
        )

        try:
            return self._parse_explanation(response)

        except Exception as e:
            print(f"Explanation parsing failed: {e}")
            return "Explanation could not be parsed."

    def _build_prompt(
        self, examples: List[Example]
    ):
        highlighted_examples = []

        for i, example in enumerate(examples):
            index = i + 1  # Start at 1
            formatted = self._highlight(index, example)

            # NOTE: Maybe add normalized activations back?
            mask = example.activations > self.activation_threshold
            activations = example.activations[mask].tolist()

            acts = ", ".join(f"({item})" for item in activations)

            formatted += "\n" + acts

            # NOTE: Using activations by default.
            highlighted_examples.append(formatted)

        highlighted_examples = "\n".join(highlighted_examples)

        return build_prompt(
            examples=highlighted_examples,
            use_cot=self.use_cot,
        )

    def _parse_explanation(self, text: str) -> str:
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

    def _highlight(self, index, example):
        result = f"Example {index}: "

        str_toks = self.tokenizer.batch_decode(example.tokens)

        activations = example.activations

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