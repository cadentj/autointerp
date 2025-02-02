import json
import asyncio

from openai import AsyncOpenAI

from ..schema import Conversation

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
                print(f"Attempt {attempt + 1}: Invalid JSON response, retrying... {e}")
            except Exception as e:
                print(f"Attempt {attempt + 1}: {str(e)}, retrying...")
            
            await asyncio.sleep(1)
            
        raise RuntimeError(f"All {self.max_retries} retry attempts failed")

    def postprocess(self, response) -> str:
        return response.choices[0].message.content
