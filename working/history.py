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
    
    def get_last_judgement(self):
        return self.history["judge"][-1]["content"]
    
    def save(self):
        import pickle
        import os 
        
        PATH = os.path.dirname(os.path.abspath(__file__))
        
        html = self.to_html()

        # get number of folders in results directory
        num_folders = len(os.listdir(PATH + "/results"))

        save_dir = PATH + f"/results/run_{num_folders}"

        os.mkdir(save_dir)

        save_path = "/results.html"
        with open(save_dir + save_path, 'w') as f:
            f.write(html)

        save_path = "/history.pkl"
        with open(save_dir + save_path, 'wb') as f:
            pickle.dump(self.history, f)
        
    def to_html(self):

        html = ""

        include_judge = False

        if "judge" in self.history:
            include_judge = True
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

            if include_judge:
                judge = judge_transcripts[round]

                html += f"<h3>Judge</h3>"
                content = judge['content'].replace('\n', '<br />')

            html += f"<p>{content}</p>"

        return html