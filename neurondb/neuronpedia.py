import aiohttp
import asyncio
from typing import List, Optional, Generator

from pydantic import model_validator, BaseModel, Field, AliasChoices, field_validator

from .config import config

TOKEN = config["token"]

######## SCHEMA ########

######## Response Models ########

class NeuronpediaActivation(BaseModel):
    compressed: bool = False
    
    tokens: List[str]

    value_pos: List[int] = []
    raw_values: List[float] = Field(
        validation_alias=AliasChoices("values", "raw_values"), 
        serialization_alias="raw_values"
    )

    @property
    def values(self) -> List[float]:
        return self.expand()

    def expand(self) -> List[float]:
        expanded_values = []
        for i in range(len(self.tokens)):
            if i in self.value_pos:
                pos = self.value_pos.index(i)
                expanded_values.append(self.raw_values[pos])
            else:
                expanded_values.append(0)
        return expanded_values
    
    @field_validator('tokens', mode='after')  
    @classmethod
    def set_whitespace(cls, tokens: List[str]) -> List[str]:
        return [t.replace("▁", " ") for t in tokens]

    @model_validator(mode='after')  
    def compress(self) -> List[float]:
        if self.compressed:
            return self
        
        value_pos = []
        raw_values = []

        for i, value in enumerate(self.raw_values):
            if value > 0:
                value_pos.append(i)
                raw_values.append(value)

        self.value_pos = value_pos
        self.raw_values = raw_values
        self.compressed = True

        return self

class NeuronpediaResponse(BaseModel):
    layer_id: str = Field(validation_alias=AliasChoices("layer", "layer_id"))
    index: int
    activations: List[NeuronpediaActivation]
    max_activation: float = Field(alias=AliasChoices("max_activation", "maxActApprox"))

    # Positive Logits
    pos_str: Optional[List[str]] = None
    pos_values: Optional[List[float]] = None

    # Negative Logits
    neg_str: Optional[List[str]] = None
    neg_values: Optional[List[float]] = None

######## Request Models ########

class NeuronpediaDictionary(BaseModel):
    layer_id: str
    indices: List[int]

class NeuronpediaRequest(BaseModel):
    model_id: str
    dictionaries: List[NeuronpediaDictionary]

    FEATURE_REQUEST_URL: str = "https://www.neuronpedia.org/api/feature/{model_id}/{layer_id}/{index}"

    def get_requests(self) -> Generator[str, None, None]: 
        for dictionary in self.dictionaries:
            for index in dictionary.indices:
                yield self.FEATURE_REQUEST_URL.format(
                    model_id=self.model_id,
                    layer_id=dictionary.layer_id,
                    index=index
                )

######## METHODS ########

async def fetch_feature(
    session: aiohttp.ClientSession,
    url: str,
) -> NeuronpediaResponse:
    headers = {'X-Api-Key': TOKEN}
    async with session.get(url, headers=headers) as response:
        if response.status == 200:
            response_json = await response.json()
            print(response_json.keys())
            return NeuronpediaResponse(**response_json)
        else:
            print(f"Error fetching feature at URL {url}: {response.status}")
            return None

async def fetch_all_features(request: NeuronpediaRequest) -> List[NeuronpediaResponse]:
    request = NeuronpediaRequest(**request)
    async with aiohttp.ClientSession() as session:
        tasks = []
        for url in request.get_requests():
            task = fetch_feature(session, url)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        return results