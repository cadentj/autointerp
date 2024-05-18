from .base import Environment

from .debater import Debater
from typing import List
from .history import History

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

    def execute_round(
        self, 
        round: int
    ):
        threads = []

        add = lambda id, response: self.history.add(id, response, round)

        prompts = self.build_round_prompts(round)

        for i, d in enumerate(self.agents):
            thread = threading.Thread(
                target=d.execute,
                args=(prompts[d.id], {}, add),
            )
            threads.append(thread)
            thread.start()
        
        for i, t in enumerate(threads):
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
                    other_responses
                )

        return prompts
            

