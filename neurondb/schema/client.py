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

### RESPONSE OBJECT ###

class LogProbs(BaseModel):
    tokens: List[str]
    token_logprobs: List[float]

class Choice(BaseModel):
    message: Message
    logprobs: LogProbs | None = None

class Response(BaseModel):
    choices: List[Choice]