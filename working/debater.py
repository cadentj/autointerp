from .base import Agent

from .utils.prompting import Client

class Debater(Agent):  
    
    def __init__(
        self,
        client: Client, 
        id: int,
    ):
        super().__init__(client)
        self.id = id

    def execute(
        self,
        prompt: str, 
        generation_args: dict,
        add: callable
    ):
        response = self.client.generate(
            prompt, 
            **generation_args
        )

        add(self.id, response)

