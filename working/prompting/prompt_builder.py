import random

from ..history import History
from ..tools import Quote
from .judged_prompts import judge_prompt


def format_list(
    l: list,
    leading: str = "",
) -> str:
    formatted_str = ""
    for i, item in enumerate(l):
        formatted_str += f"{leading} {i}: {item}\n"
    return formatted_str


class PromptBuilder():

    def __init__(
        self,
        debate,
        top_examples,
        top_logits
    ):
        self.history = debate.history
        self.debate = debate

        self.top_examples = [example.text for example in top_examples]
        self.top_logits = top_logits
        self.quote = Quote(self.top_examples)

    def build_history(
        self,
        opening_prompt: str,
        split: bool = False,
        seed: int = 22
    ):
        random.seed(seed)
        
        if split:
            random.shuffle(self.top_examples)

            n_debaters = len(self.debate.debaters)
            split_size = len(self.top_examples) // n_debaters
            split_examples = [
                self.top_examples[i:i+split_size] for i in range(0, len(self.top_examples), split_size)
            ]

            for i, debater in enumerate(self.debate.debaters):
                prompt = opening_prompt.format(
                    examples = format_list(split_examples[i], leading=f"Example"),
                    top_logits = f"Top_logits: {self.top_logits}"
                )

                turn = {
                    "role" : "user",
                    "content" : prompt
                }

                self.history.add(debater.id, turn)
                self.history.add_examples(debater.id, split_examples[i])

        else: 

            for debater in self.debate.debaters:

                random.shuffle(self.top_examples)

                prompt = opening_prompt.format(
                    examples = format_list(self.top_examples, leading=f"Example")
                )

                turn = {
                    "role" : "user",
                    "content" : prompt
                }

                self.history.add(debater.id, turn)
            
            self.history.add_examples("all", self.top_examples)
        

    def parse_argument(self, debater):
        response = self.history.history[debater.id][-1]['content']
        argument = response.split("[/THOUGHTS]")[-1]

        verified_argument = self.quote(argument)
        return verified_argument

    def build_judge_prompt(
        self,
    ):
        arguments = []

        for d in self.debate.debaters:
            argument = self.parse_argument(d)
            arguments.append(argument)

        turn = {
            "role" : "user",
            "content" : judge_prompt.format(
                debater_arguments=format_list(arguments, "Arguments"),
            )
        }

        self.history.add("judge", turn)

    def parse_others(self, other_responses):
        parsed_others = ""
        for agent, response in other_responses.items():
            parsed_others += f"[{agent} RESPONSE]:\n{response}\n"
    
    def build_debater_prompt(
        self,
        round_start_prompt: str
    ):
        for debater in self.debate.debaters:
            other_responses = self.history.get_other_responses(debater.id)

            if "judge" in self.history.history:
                judge_evaluation = self.history.get_last_judgement()

                prompt = round_start_prompt.format(
                    other_responses=other_responses,
                    judge_evaluation=judge_evaluation
                )
            else:
                prompt = round_start_prompt.format(
                    other_responses=other_responses
                )

            turn = {
                "role" : "user",
                "content" : prompt
            }

            self.history.add(debater.id, turn)