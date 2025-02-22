from neurondb.autointerp import OpenRouterClient
import asyncio
import requests 


PROVIDER_KWARGS = {
    "provider": {
        "order": [
            "Lambda"
        ]
    }
}

import requests
import json




GENERATION_KWARGS = {
    "logprobs": 1,
    "provider" : {
        "order" : [
            "Together"
        ]
    }
}

client = OpenRouterClient(model="meta-llama/Llama-3.3-70B-Instruct-Turbo")
async def test():
    response = await client.generate([
        {
            "role": "user",
            "content": "What is 1010 + 19 + 034"
        }
    ], **GENERATION_KWARGS)
    print(response.model_dump()) 

    print(response.choices[0].logprobs)

if __name__ == "__main__":
    asyncio.run(test())