from typing import List

from .debate import Debate
from ..agents import Debater
from ..prompting import PromptBuilder

from ..prompting.collaborative_prompts import opening_prompt, round_start_prompt

class CollabDebate(Debate):
    
    def __init__(
        self,
        debaters: List[Debater],
    ):
        
        self.history = None
        self.debaters = None

        super().__init__(debaters)

        self.prompt_builder = PromptBuilder(self)
        self.prompt_builder.build_history(opening_prompt)
    
    def run(
        self,
        max_rounds: int = 3,
    ):
        for _ in range(max_rounds):
            self.debate()
            self.prompt_builder.build_debater_prompt(
                round_start_prompt
            )

        self.history.save()        

    

