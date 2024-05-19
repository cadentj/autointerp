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
    ) -> str:
        
        last_round = {
            id: response[-1]["content"]
            for id, response in self.history.items()
        }

        last_round.pop(id)
        return last_round
    
    def to_html(self, save_path: str):

        html = ""
        
        judge_transcripts = self.history["judge"][1::2]
        debater_transcripts = [
            value[1::2] for key, value in self.history.items() 
            if key != "judge"
        ]
        
        for round, debaters in enumerate(zip(*debater_transcripts)):
            
            html += f"<h2>Round {round}</h2>"
            html += f"<div style='display: flex;'>"

            for i, debater in enumerate(debaters):
                if debater['role'] == "user":
                    continue

                html += f"<div style='flex: 1; padding: 10px;'>"
                html += f"<h3>Debater {i}</h3>"
                content = debater['content'].replace('\n', '<br />')
                html += f"<p>{content}</p>"

                html += "</div>"

            html += "</div>"

            html += f"<div style='padding: 10px;'>"

            judge = judge_transcripts[round]

            html += f"<h3>Judge</h3>"
            content = judge['content'].replace('\n', '<br />')
            html += f"<p>{content}</p>"

        with open(save_path, 'w') as f:
            f.write(html)