import aiohttp
import asyncio
import os
from typing import List

from .schema import DictionaryRequest, NeuronpediaResponse


FEATURE_REQUEST_URL = (
    "https://www.neuronpedia.org/api/feature/{model_id}/{layer_id}/{index}"
)
# TOKEN = os.environ["NEURONPEDIA_TOKEN"]
TOKEN = ""


async def fetch_feature(
    session: aiohttp.ClientSession,
    url: str,
) -> NeuronpediaResponse:
    headers = {"X-Api-Key": TOKEN}
    async with session.get(url, headers=headers) as response:
        if response.status == 200:
            response_json = await response.json()
            return NeuronpediaResponse(**response_json)
        else:
            print(f"Error fetching feature at URL {url}: {response.status}")
            return None


async def load_neuronpedia(
    model_id: str,
    dictionaries: DictionaryRequest,
) -> List[NeuronpediaResponse]:
    async with aiohttp.ClientSession() as session:
        tasks = []
        for layer_id, indices in dictionaries.items():
            for index in indices:
                url = FEATURE_REQUEST_URL.format(
                    model_id=model_id, layer_id=layer_id, index=index
                )
                task = fetch_feature(session, url)
                tasks.append(task)

        results = await asyncio.gather(*tasks)

    return results
