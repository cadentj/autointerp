import re
from typing import Literal
import math


from .prompts.explainer_prompt import build_prompt
from openai import AsyncOpenAI
from ..base import Feature


class Query:
    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
        formatted_query_prompt: str,
        max_or_min: Literal["max", "min"] = "max",
        threshold: float = 0.0,
        insert_as_prompt: bool = False,
        verbose: bool = False,
    ):
        self.client = client
        self.model = model
        self.verbose = verbose
        self.insert_as_prompt = insert_as_prompt
        self.formatted_query_prompt = formatted_query_prompt
        
        self.max_or_min = max_or_min
        self.threshold = threshold

    async def __call__(
        self,
        feature: Feature,
        **generation_kwargs,
    ):
        messages = self._build_prompt(feature)

        explanation_completion = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=200,
            temperature=0,
            seed=0,
            **generation_kwargs,
        )
        explanation_response = explanation_completion.choices[0].message.content
        explanation = self._parse_explanation(explanation_response)

        # Include the query prompt
        messages.extend([
            {
                "role": "assistant",
                "content": explanation_response,
            },
            {
                "role": "user",
                "content": self.formatted_query_prompt,
            },
        ])

        completion = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=1,
            temperature=0,
            logprobs=True,
            top_logprobs=20,
            seed=0,
            **generation_kwargs,
        )

        logprobs = self.logprob_probs(completion)
        score = self._aggregate_0_100_score(logprobs)

        return explanation, score

    def logprob_probs(self, completion) -> dict:
        """Simple logprobs request. Returns probabilities. Always samples 1 token."""
        try:
            logprobs = completion.choices[0].logprobs.content[0].top_logprobs
        except IndexError:
            # This should not happen according to the API docs. But it sometimes does.
            return {}

        result = {}
        for el in logprobs:
            result[el.token] = float(math.exp(el.logprob))

        return result

    def _aggregate_0_100_score(self, score: dict) -> float:
        #   NOTE: we don't check for refusals explcitly. Instead we assume that
        #   if there's at least 0.25 total weight on numbers, it's not a refusal.
        total = 0
        sum_ = 0
        for key, val in score.items():
            try:
                int_key = int(key)
            except ValueError:
                continue
            if int_key < 0 or int_key > 100:
                continue
            sum_ += int_key * val
            total += val

        if total < 0.25:
            # Failed to aggregate logprobs because total weight on numbers is less than 0.25.
            return None
        return sum_ / total
    
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
