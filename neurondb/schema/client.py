from typing import List, NamedTuple

from pydantic import BaseModel
from torchtyping import TensorType

class Message(BaseModel):
    role: str
    content: str

class Conversation(BaseModel):
    messages: List[Message]


class PromptLogProbs(NamedTuple):
    indices: TensorType["seq", "top_k"]
    values: TensorType["seq", "top_k"]
    tokens: TensorType["seq"]