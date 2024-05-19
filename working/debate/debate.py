import threading
from typing import List
from functools import partial

from ..utils import format_list
from ..agents import Debater
from ..agents import Judge
from ..history import History
from ..prompts import opening_prompt, round_start_prompt, examples

class Debate():
    
    def __init__(
        self,
        debaters: List[Debater],
        judge: Judge
    ):
        
        self.debaters = debaters
        self.judge = judge

        self.history = self.create_history()

    def create_history(self):
        history = History()

        prompt = opening_prompt.format(
            examples=examples
        )

        turn = {
            "role" : "user",
            "content" : prompt
        }

        for debater in self.debaters:
            history.add(debater.id, turn)

        return history

    def run(
        self,
        max_rounds: int = 3
    ):
        for _ in range(max_rounds):
            self.debate()
            # self.judge()
        
    def debate(self):
        threads = []

        for d in self.debaters:
            history_add = partial(self.history.add, d.id)

            thread = threading.Thread(
                target=d.execute,
                args=(
                    self.history.history[d.id], 
                    {}, 
                    lambda turn, add=history_add: add(turn)
                ),
            )
            
            threads.append(thread)
            thread.start()

        for t in threads:
            t.join()


    def judge(self):
        self.judge.execute(
            self.history['judge'],
            {},
            lambda turn: self.history.add("judge", turn)
        )

    

