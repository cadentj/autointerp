from .base import Agent

from .utils.prompting import Client

class Judge(Agent):  
    
    def __init__(
        self,
        client: Client,
    ):
        super().__init__(client)

    def __call__(
        self,
        prompt: str, 
        generation_args: dict,
        add: callable
    ):
        response = self.client.generate(
            prompt
        )

        turn = {
            "user": prompt,
            "assistant": response
        }

        add(turn)

    

