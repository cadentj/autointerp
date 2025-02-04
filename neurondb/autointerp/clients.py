import json
import asyncio

from openai import AsyncOpenAI

import torch as t
from typing import List
from nnsight import LanguageModel
from ..schema import Conversation
from ..schema.client import PromptProbs
from ..utils import load_tokenizer


class LocalClient:
    def __init__(
        self, model: str, base_url="http://localhost:8000/v1", max_retries=2
    ):
        self.client = AsyncOpenAI(
            base_url=base_url, api_key="EMPTY", timeout=None
        )
        self.max_retries = max_retries
        self.model = model

    async def generate(self, messages: Conversation, **kwargs):
        for attempt in range(self.max_retries):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model, messages=messages, **kwargs
                )

                if response is None:
                    raise ValueError("Response is None")

                return self.postprocess(response)

            except json.JSONDecodeError as e:
                print(
                    f"Attempt {attempt + 1}: Invalid JSON response, retrying... {e}"
                )
            except Exception as e:
                print(f"Attempt {attempt + 1}: {str(e)}, retrying...")

            await asyncio.sleep(1)

        raise RuntimeError(f"All {self.max_retries} retry attempts failed")

    def postprocess(self, response) -> str:
        return response.choices[0].message.content


class NsClient:
    def __init__(self, model_id: str, k=15, **model_kwargs):
        model = LanguageModel(
            model_id, device_map="auto", dispatch=True, **model_kwargs
        )
        tokenizer = load_tokenizer(model_id)
        model.tokenizer = tokenizer

        self.k = k
        self.model = model
        self.tokenizer = tokenizer

        try:
            _ = self.model.lm_head
        except Exception:
            raise ValueError("Client not compatible with this model.")

    def _prepare_input(self, conversations: List[Conversation]):
        for conversation in conversations:
            if conversation[-1]["role"] != "assistant":
                raise ValueError("Last message must be an assistant message")

        inputs = self.tokenizer.apply_chat_template(
            conversations,
            tokenize=True,
            return_assistant_tokens_mask=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
            truncation=False
        )

        assistant_tokens_mask = t.tensor(inputs["assistant_masks"]) == 1
        inputs["assistant_masks"] = assistant_tokens_mask

        return inputs

    def _prepare_response(self, inputs, assistant_logits) -> List[PromptProbs]:
        log_probs = []
        for batch_idx, topk_probs in enumerate(assistant_logits):
            assistant_tokens_mask = inputs["assistant_masks"][batch_idx]
            tokens = inputs["input_ids"][batch_idx][assistant_tokens_mask]
            log_probs.append(
                PromptProbs(
                    indices = topk_probs.indices,
                    values = topk_probs.values,
                    tokens=tokens,
                )
            )

        return log_probs

    def generate(self, conversations: List[Conversation]) -> List[PromptProbs]:
        inputs = self._prepare_input(conversations)

        assistant_logits = []
        with self.model.trace(inputs):
            logits = self.model.lm_head.output

            for batch_idx, conversation_mask in enumerate(
                inputs["assistant_masks"]
            ):
                logits_slice = logits[batch_idx][conversation_mask]
                probs = logits_slice.softmax(dim=-1)
                top_probs = probs.topk(self.k, dim=-1)
                probs = top_probs.save()
                assistant_logits.append(probs)

        response = self._prepare_response(inputs, assistant_logits)
        return response
