from .base import Environment

from .debater import Debater
from .judge import Judge
from typing import List
from .history import History
from functools import partial
import threading

from .prompts import opening_prompt, round_start_prompt


examples = """Example 1: reconstruction��, but because I��<<m>> evidence-based and
Example 2: and innovate on until we create something that we all appreciate. I��<<m>> excited to see
what
Example 3: of the May 1968 graffitists wrote: ��I��<<m>> not a servant of
Example 4: the city,�� Rep. Jackson said. ��I��<<m>> sure the mayor is
Example 5: bad year in Augusta, I said to myself, ��I��<<m>> sick of this.
Example 6: Nothing against those that hold belts, and fight big shows. I��<<m>> sure those achievements
are
Example 7: , right? How are you approaching combat in this one? I��<<m>> guessing you��
Example 8: ocks. It was a pretty high-achieving school. I��<<m>> not"""

class Debate(Environment):
    
    def __init__(
        self,
        debaters: List[Debater],
        judge: Judge
    ):
        
        self.agents = None
        self.judge = judge

        self.history = History()

        super().__init__(debaters)


    def run(
        self,
        max_rounds: int = 3
    ):
        for round in range(max_rounds):
            self.execute_round(round)


    def execute_round(self, round: int):
        prompts = self.build_prompts(round)

        self.execute_debaters(prompts)
        self.execute_judge(prompts)
        

    def execute_debaters(self, prompts):
        threads = []

        for d in self.agents:
            # Partial creates a new function
            # so threading doesn't break w reference.
            add_history = partial(self.history.add, d.id, round=round)

            thread = threading.Thread(
                target=d.execute,
                args=(
                    prompts[d.id], 
                    {}, 
                    lambda turn, add=add_history: add(turn)
                ),
            )
            
            threads.append(thread)
            thread.start()

        for t in threads:
            t.join()

    def execute_judge(self, prompts):
        
        self.judge(
            prompts['judge'],
            {},
            lambda turn: self.history.add("judge", turn, round)
        )

    def build_prompts(
        self,
        round: int
    ):
        prompts = {}

        for d in self.agents:
            if round == 0:
                prompts[d.id] = [opening_prompt.format(
                    examples=examples
                )]
            else:
                prompts[d.id].append(
                    round_start_prompt.format(
                        self.create_response_list(d),
                        judge_evaluation=self.history.get_judge_evaluation(round)
                    )
                )
        
        prompts["judge"] = [""]

    def create_response_list(
        self,
        debater: Debater,
    ):
        other_responses = self.history.get_other_responses(debater.id)

        response_str = ""
        for id, response in other_responses.items():
            response_str += f"{id}: {response["assistant"]}\n"
        
        return response_str

            

