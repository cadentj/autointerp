import aiohttp
import asyncio
import os
import json
from typing import List, Optional, Dict, Literal

from pydantic import (
    model_validator,
    BaseModel,
    Field,
    AliasChoices,
    field_validator,
)


class NeuronpediaExample(BaseModel):
    compressed: bool = False

    tokens: List[str]

    value_pos: List[int] = []
    raw_values: List[float] = Field(
        validation_alias=AliasChoices("values", "raw_values"),
        serialization_alias="raw_values",
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

    @field_validator("tokens", mode="after")
    @classmethod
    def set_whitespace(cls, tokens: List[str]) -> List[str]:
        return [t.replace("▁", " ") for t in tokens]

    @model_validator(mode="after")
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

    activations: List[NeuronpediaExample]
    max_activation: float = Field(
        alias=AliasChoices("max_activation", "maxActApprox")
    )

    # Positive Logits
    pos_str: Optional[List[str]] = None
    pos_values: Optional[List[float]] = None

    # Negative Logits
    neg_str: Optional[List[str]] = None
    neg_values: Optional[List[float]] = None

    def to_html(
        self,
        threshold: float = 0.0,
        n: int = 10,
    ) -> str:
        def _to_html(
            tokens: List[str], activations: List[float]
        ) -> str:
            result = []
            max_act = max(activations)
            _threshold = max_act * threshold

            for i in range(len(tokens)):
                if any(t in tokens[i] for t in ["<", ">", "/"]):
                    result.append("HTML_SKIP")
                    continue

                if activations[i] > _threshold:
                    # Calculate opacity based on activation value (normalized between 0.2 and 1.0)
                    opacity = 0.2 + 0.8 * (activations[i] / max_act)
                    result.append(
                        f'<mark style="opacity: {opacity:.2f}">{tokens[i]}</mark>'
                    )
                else:
                    result.append(tokens[i])

            return "".join(result)

        strings = [
            _to_html(example.tokens, example.values)
            for example in self.activations[:n]
        ]

        return "<br><br>".join(strings)

    def display(
        self,
        threshold: float = 0.0,
        n: int = 10,
    ) -> None:
        from IPython.display import HTML, display

        display(HTML(self.to_html(threshold, n)))


class NeuronpediaCache:
    def __init__(self, features: List[NeuronpediaResponse]):
        self.features = features

    def save_to_disk(self, path: str):
        with open(path, "w") as f:
            json.dump([feature.model_dump() for feature in self.features], f)

    @classmethod
    def load_from_disk(cls, path: str) -> "NeuronpediaCache":
        with open(path, "r") as f:
            return cls(
                [NeuronpediaResponse(**feature) for feature in json.load(f)]
            )

    def save_to_html(self, path: str) -> None:
        html = []
        title = "<h2>Layer {layer_id}, Index {index}</h2>"
        for feature in self.features:
            html.append(
                title.format(layer_id=feature.layer_id, index=feature.index)
            )
            html.append(feature.to_html())

        with open(path, "w") as f:
            f.write("<br>".join(html))


Models = Literal["gemma-2-2b", "gemma-2-9b"]
FEATURE_REQUEST_URL = (
    "https://www.neuronpedia.org/api/feature/{model_id}/{layer_id}/{index}"
)
TOKEN = os.environ.get("NEURONPEDIA_TOKEN", None)


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
    model_id: Models,
    dictionaries: Dict[int, List[int]],
) -> NeuronpediaCache:
    print("Only 16k SAEs supported.")
    async with aiohttp.ClientSession() as session:
        tasks = []
        for layer_id, indices in dictionaries.items():
            _layer_id = f"{layer_id}-gemmascope-res-16k"
            for index in indices:
                url = FEATURE_REQUEST_URL.format(
                    model_id=model_id, layer_id=_layer_id, index=index
                )
                task = fetch_feature(session, url)
                tasks.append(task)

        results = await asyncio.gather(*tasks)

    return NeuronpediaCache(results)


def sync_load_neuronpedia(
    model_id: Models,
    dictionaries: Dict[int, List[int]],
    patch_notebook: bool = False,
) -> NeuronpediaCache:
    if patch_notebook:
        import nest_asyncio

        nest_asyncio.apply()

    return asyncio.run(load_neuronpedia(model_id, dictionaries))
