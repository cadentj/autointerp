from collections import defaultdict

class History():

    def __init__(self):
        self.history = defaultdict(list)
        self.top_examples = {}

    def add(
        self,
        id: str,
        turn: dict,
    ) -> None:
        self.history[id].append(turn)

    def add_examples(
        self,
        id: str,
        examples,
    ) -> None:
        self.top_examples[id] = examples

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
    
    def save(self, provider, path):
        import pickle
        import os 
        
        PATH = os.path.dirname(os.path.abspath(__file__))
        
        html = self.to_html()

        save_dir = PATH + f"/results/{provider}/{path}"

        os.mkdir(save_dir)

        save_path = "/results.html"
        with open(save_dir + save_path, 'w') as f:
            f.write(html)

        save_path = "/history.pkl"
        with open(save_dir + save_path, 'wb') as f:
            pickle.dump(self.history, f)
        
    def highlight_example(self, top_examples):
        html_content = ""
        for example in top_examples:
            formatted_example = example.replace("<<", "<mark>").replace(">>", "</mark>")
            html_content += f'<p>{formatted_example}</p>'
        
        return html_content

    def to_html(self):

        html = ""

        include_judge = False

        if "judge" in self.history:
            include_judge = True
            judge_transcripts = self.history["judge"][1::2]

            html += f"<h2>Top Examples</h2>"
            top_examples = self.highlight_example(self.top_examples['all'])
            html += top_examples
        else:
            html += f"<h2>Top Examples</h2>"

            for debater, examples in self.top_examples.items():
                html += f"<h3>Debater {debater}</h3>"
                top_examples = self.highlight_example(examples)
                html += top_examples

        
        debater_transcripts = [
            value[1::2] for key, value in self.history.items() 
            if key != "judge"
        ]
        
        for round, debaters in enumerate(zip(*debater_transcripts)):
            
            html += f"<h2>Round {round + 1}</h2>"
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

            if include_judge:
                html += f"<div style='padding: 10px;'>"
                judge = judge_transcripts[round]

                html += f"<h3>Judge</h3>"
                content = judge['content'].replace('\n', '<br />')

                html += f"<p>{content}</p>"
                html += "</div>"

        return html