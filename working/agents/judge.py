from . import Agent

from ..utils.prompting import Client

class Judge(Agent):  
    
    def __init__(
        self,
        client: Client,
    ):
        super().__init__(client)

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

    

