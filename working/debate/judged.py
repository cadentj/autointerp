from typing import List

from .debate import Debate
from ..agents import Debater
from ..agents import Judge
from ..prompting import PromptBuilder

from ..prompting.judged_prompts import opening_prompt, round_start_prompt

class JudgedDebate(Debate):
    
    def __init__(
        self,
        debaters: List[Debater],
        judge: Judge
    ):
        
        self.history = None
        self.debaters = None

        super().__init__(debaters)

        self.judge = judge

        self.prompt_builder = PromptBuilder(self)

        self.prompt_builder.build_history(opening_prompt)
    
    def run(
        self,
        max_rounds: int = 3,
    ):
        for _ in range(max_rounds):
            self.debate()
            self.prompt_builder.build_judge_prompt()
            self.judge_round()
            self.prompt_builder.build_debater_prompt(
                round_start_prompt
            )

        self.history.save()        
        
    def judge_round(self):

        prompt = [self.history.history['judge'][-1]]

        self.judge.execute(
            prompt,
            {},
            lambda turn: self.history.add("judge", turn)
        )

    

