import threading
from typing import List
from functools import partial

from ..agents import Debater
from ..agents import Judge
from ..history import History
from ..utils import PromptBuilder

class Debate():
    
    def __init__(
        self,
        debaters: List[Debater],
        judge: Judge
    ):
        
        self.history = History()
        self.debaters = debaters
        self.judge = judge

        self.prompt_builder = PromptBuilder(self)
        
        self.prompt_builder.build_history()

    def run(
        self,
        max_rounds: int = 3,
    ):
        for _ in range(max_rounds):
            self.debate()
            self.prompt_builder.build_judge_prompt()
            self.judge_round()
            self.prompt_builder.build_debater_prompt()

        self.history.save()        
        
    def debate(self):
        threads = []

        for d in self.debaters:
            history_add = partial(self.history.add, d.id)

            thread = threading.Thread(
                target=d.execute,
                args=(
                    self.history.history[d.id], # this is turn
                    {}, 
                    lambda turn, add=history_add: add(turn)
                ),
            )
            
            threads.append(thread)
            thread.start()

        for t in threads:
            t.join()


    def judge_round(self):

        prompt = [self.history.history['judge'][-1]]

        self.judge.execute(
            prompt,
            {},
            lambda turn: self.history.add("judge", turn)
        )

    

