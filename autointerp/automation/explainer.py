import re
from typing import List

from .prompts.explainer_prompt import build_prompt
from .clients import HTTPClient
from ..base import Example


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

    async def __call__(self, examples: List[Example], **generation_kwargs):
        messages = self._build_prompt(examples)

        response = await self.client.generate(messages, **generation_kwargs)

        if self.verbose:
            with open("response.txt", "w") as f:
                for message in messages:
                    f.write(f"{message['role'].upper()}:\n\n")
                    f.write(message["content"] + "\n\n")
                f.write(response)

        try:
            return self._parse_explanation(response)

        except Exception as e:
            print(f"Explanation parsing failed: {e}")
            return "Explanation could not be parsed."

    def _build_prompt(self, examples: List[Example]):
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
