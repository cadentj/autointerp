from . import Agent

from ..utils.api import Client

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
            prompt
        )

        turn = {
            "role": "assistant",
            "content": response
        }

        add(turn)

    

