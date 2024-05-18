from .base import Environment

from .debater import Debater
from typing import List
from .history import History
from functools import partial
import threading

from .prompts import opening_prompt, round_start_prompt

class Debate(Environment):
    
    def __init__(
        self,
        debaters: List[Debater],
    ):
        
        self.agents = None
        self.history = History()

        super().__init__(debaters)

    def run(self):
        pass

    def execute_round(self, round: int):
        threads = []

        prompts = self.build_round_prompts(round)

        for d in self.agents:
            history_adder = partial(self.history.add, d.id, round=round)

            thread = threading.Thread(
                target=d.execute,
                args=(
                    prompts[d.id], 
                    {}, 
                    lambda response, adder=history_adder: adder(response)
                ),
            )
            
            threads.append(thread)
            thread.start()

        for t in threads:
            t.join()

    def build_round_prompts(
        self,
        round: int
    ):
        prompts = {}
        for d in self.agents:
            if round == 0:
                prompts[d.id] = opening_prompt

            else:
                other_responses = self.history.get_other_responses(
                    d.id,
                    round - 1
                )

                prompts[d.id] = round_start_prompt.format(
                    other_responses=other_responses
                )

        return prompts
    
    
            

