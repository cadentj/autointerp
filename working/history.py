from collections import defaultdict

class History():

    def __init__(self):
        self.history = defaultdict(list)

    def add(
        self,
        id: str,
        turn: dict,
    ) -> None:
        self.history[id].append(turn)

    def get_other_responses(
        self,
        id: str,
        round: int
    ) -> str:
        other_responses = self.get_round(round)
        other_responses.pop(id)
        return other_responses
    
    def get_judge(
        self,
    ) -> str:
        return self.history["judge"]

    def get_round(
        self,
        round: int,
    ) -> list:
        return {
            id: response[round] 
            for id, response in self.history.items()
        }
    