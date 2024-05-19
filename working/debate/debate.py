import threading
from typing import List
from functools import partial

from ..utils import format_list
from ..agents import Debater
from ..agents import Judge
from ..history import History
from ..prompts import opening_prompt, first_round_start_prompt, judge_prompt, examples

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
        for _ in range(2):
            self.debate()

            self.build_prompts()

            self.judge_round()


    def build_prompts(self):

        arguments = []

        for d in self.debaters:
            
            # Parse the argument from the last response
            response = self.history.history[d.id][-1]['content']
            argument = response.split("[/THOUGHTS]")[-1]
            arguments.append(argument)

            # Load the next prompt into history
            other_responses = self.history.get_other_responses(d.id)

            prompt = first_round_start_prompt.format(
                other_responses=other_responses
            )

            turn = {
                "role" : "user",
                "content" : prompt
            }

            self.history.add(d.id, turn)

        turn = {
            "role" : "user",
            "content" : judge_prompt.format(
                debater_arguments=format_list(arguments, "Arguments"),
            )
        }

        self.history.add("judge", turn)
        
        
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

    

