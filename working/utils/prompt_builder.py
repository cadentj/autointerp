from ..history import History
from ..tools import Quote
from ..utils.prompts import opening_prompt, round_start_prompt, judge_prompt, examples


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
        top_examples: str = examples
    ):

        self.history = debate.history
        self.debate = debate
        self.quote = Quote(top_examples)

    def build_history(self):

        prompt = opening_prompt.format(
            examples=examples
        )

        turn = {
            "role" : "user",
            "content" : prompt
        }

        for debater in self.debate.debaters:
            self.history.add(debater.id, turn)
        

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
    
    def build_debater_prompt(self):
        for debater in self.debate.debaters:
            other_responses = self.history.get_other_responses(debater.id)
            judge_evaluation = self.history.get_last_judgement()

            prompt = round_start_prompt.format(
                other_responses=other_responses,
                judge_evaluation=judge_evaluation
            )

            turn = {
                "role" : "user",
                "content" : prompt
            }

            self.history.add(debater.id, turn)