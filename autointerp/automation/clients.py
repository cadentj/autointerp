import asyncio
import json
import os
from typing import List, NamedTuple

from anthropic import Anthropic
from anthropic.types.message_create_params import (
    MessageCreateParamsNonStreaming,
)
from anthropic.types.messages.batch_create_params import Request
import httpx
from pydantic import BaseModel
import torch as t
from torchtyping import TensorType
from transformers import AutoModelForCausalLM

from ..utils import load_tokenizer

### SCHEMA ###

class Message(BaseModel):
    role: str
    content: str

class Conversation(BaseModel):
    messages: List[Message]

class PromptLogProbs(NamedTuple):
    indices: TensorType["seq", "top_k"]
    values: TensorType["seq", "top_k"]
    tokens: TensorType["seq"]

class LogProbs(BaseModel):
    tokens: List[str]
    token_logprobs: List[float]

class Choice(BaseModel):
    message: Message
    # logprobs: LogProbs | None = None

class Response(BaseModel):
    choices: List[Choice]


### CLIENTS ###

class HTTPClient:
    def __init__(
        self,
        model: str,
        base_url: str,
        max_retries: int,
        api_key: str,
        client=None,
    ):
        if client is None:
            self.client = httpx.AsyncClient()
        else:
            self.client = client
        self.headers = {"Authorization": f"Bearer {api_key}"}
        self.url = base_url
        self.max_retries = max_retries
        self.model = model

    async def generate(
        self, messages: Conversation, raw: bool = False, **kwargs
    ):
        for attempt in range(self.max_retries):
            try:
                response = await self.client.post(
                    url=self.url,
                    headers=self.headers,
                    json={
                        "model": self.model,
                        "messages": messages,
                        **kwargs,
                    },
                )

                if response is None:
                    raise ValueError("Response is None")

                response = Response(**response.json())

                if raw:
                    return response
                else:
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

# NOTE: Not tested with httpx
class LocalClient(HTTPClient):
    def __init__(self, model: str, max_retries=2):
        super().__init__(
            model, "http://localhost:8000/v1", max_retries, "EMPTY"
        )


class OpenRouterClient(HTTPClient):
    def __init__(self, model: str, max_retries=2):
        api_key = os.environ.get("OPENROUTER_KEY")
        if api_key is None:
            raise ValueError("OPENROUTER_KEY is not set")

        super().__init__(
            model, "https://openrouter.ai/api/v1/chat/completions", max_retries, api_key
        )


class AnthropicClient:
    def __init__(self, model: str, max_retries: int = 2):
        self.client = Anthropic()
        self.max_retries = max_retries
        self.model = model

        self.requests = []

    def _add_cache_control(self, messages: Conversation):
        # Set second to last message to cache control
        content = messages[-2]["content"]
        messages[-2]["content"] = [
            {
                "type": "text",
                "text": content,
                "cache_control": {"type": "ephemeral"},
            }
        ]

        return messages

    def upload(self):
        batch = self.client.messages.batches.create(requests=self.requests)

        return batch

    async def generate(self, messages: Conversation, **kwargs):
        feature_id = kwargs.pop("feature_id")
        messages = self._add_cache_control(messages)

        request = Request(
            custom_id=feature_id,
            params=MessageCreateParamsNonStreaming(
                model=self.model, messages=messages, **kwargs
            ),
        )

        self.requests.append(request)

        return "\\boxed{Added request for " + feature_id + "}"


class NsClient:
    def __init__(self, model_id: str, k=15, **model_kwargs):
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            attn_implementation="flash_attention_2",
            **model_kwargs,
        )
        tokenizer = load_tokenizer(model_id)

        self.k = k
        self.model = model
        self.tokenizer = tokenizer

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
            truncation=False,
        )

        assistant_mask = inputs.pop("assistant_masks")
        assistant_tokens_mask = t.tensor(assistant_mask) == 1

        inputs = inputs.to(self.model.device)

        return inputs, assistant_tokens_mask

    def _prepare_response(
        self, inputs, assistant_logits, assistant_tokens_mask
    ) -> List[PromptLogProbs]:
        log_probs = []
        for batch_idx, topk_probs in enumerate(assistant_logits):
            assistant_mask = assistant_tokens_mask[batch_idx]
            tokens = inputs["input_ids"][batch_idx][assistant_mask]
            log_probs.append(
                PromptLogProbs(
                    indices=topk_probs.indices.tolist(),
                    values=topk_probs.values.tolist(),
                    tokens=tokens.tolist(),
                )
            )

        return log_probs

    def generate(
        self, conversations: List[Conversation]
    ) -> List[PromptLogProbs]:
        inputs, assistant_tokens_mask = self._prepare_input(conversations)

        assistant_logits = []
        logits = self.model(**inputs).logits

        for batch_idx, conversation_mask in enumerate(assistant_tokens_mask):
            logits_slice = logits[batch_idx][conversation_mask]
            probs = logits_slice.log_softmax(dim=-1)
            top_probs = probs.topk(self.k, dim=-1)
            assistant_logits.append(top_probs)

        response = self._prepare_response(
            inputs, assistant_logits, assistant_tokens_mask
        )
        return response
