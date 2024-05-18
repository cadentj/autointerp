from .base import Agent

from .utils.prompting import Client

class Debater(Agent):  
    
    def __init__(
        self,
        client: Client, 
    ):
        super().__init__(client)

    def execute(
        self,
        prompt: str, 
        generation_args: dict
    ) -> str:
        response = self.client.generate(
            prompt, 
            **generation_args
        )