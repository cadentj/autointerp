from typing import List, Optional, Dict, NamedTuple

from torchtyping import TensorType
from pydantic import (
    model_validator,
    BaseModel,
    Field,
    AliasChoices,
    field_validator,
)


DictionaryRequest = Dict[str, List[int]]

class Example(NamedTuple):
    tokens: TensorType["seq"]
    activations: TensorType["seq"]

class NeuronpediaActivation(BaseModel):
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
    activations: List[NeuronpediaActivation]
    max_activation: float = Field(
        alias=AliasChoices("max_activation", "maxActApprox")
    )

    # Positive Logits
    pos_str: Optional[List[str]] = None
    pos_values: Optional[List[float]] = None

    # Negative Logits
    neg_str: Optional[List[str]] = None
    neg_values: Optional[List[float]] = None
