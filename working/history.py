from collections import defaultdict

class History():

    def __init__(self):
        self.history = defaultdict(lambda: defaultdict(dict))

    def add(
        self,
        id: str,
        turn: dict,
        round: int
    ) -> None:
        self.history[id][round] = turn

    def get_other_responses(
        self,
        id: str,
        round: int
    ) -> str:
        other_responses = self.get_round(round)
        other_responses.pop(id)
        return other_responses
    
    def get_judge_evaluation(
        self,
        round: int
    ) -> str:
        return self.get_round(round)["judge"]["assistant"]

    def get_debate_round(
        self,
        round: int
    ) -> list:
        return {id: response[round] for id, response in self.history.items() if id != "judge"}