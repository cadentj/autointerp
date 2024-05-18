from abc import ABC, abstractmethod

from ..utils.prompting import Client

class Agent(ABC):

    def __init__(
        self, 
        client: Client
    ):
        self.client = client

    @abstractmethod
    def execute(
        self,
        prompt: str, 
        generation_args: dict
    ) -> str:
        pass