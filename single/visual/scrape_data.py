# %%
import os
import re
import json
from collections import defaultdict

block = """
<h2>{feature_id}</h2>
{explanations}
"""

explanation = """
<h3>Explanation {i}</h3>
<p><b>Detection Score:</b> {d_score}<br>
<b>Generation Score:</b> {gen_score}</p>
<p>{text}</p>
"""



def get_data(text):
    max_activation = re.search(r"Max Activation: ([\d\.]+)", text).group(1)

    # Parse explanations
    pattern = r"Explanation \d+:\s*Detection Score: \(([\d\.]+), ([\d\.]+)\)\s*Generation Score: ([\d\.]+)\s*Explanation: (.+?)(?=\n\n|Explanation \d+:|$)"
    explanations = re.findall(pattern, text, re.DOTALL)

    results = {}

    for i, exp in enumerate(explanations):
        results[i] = {
            "d_score": (float(exp[0]), float(exp[1])),
            "gen_score": float(exp[2]),
            "text": exp[3]
        }

    return max_activation, results

def get_results(provider):
    results = defaultdict(dict)
    results_path = f"../results/{provider}"

    # get list of all files in the results directory
    results_files = os.listdir(results_path)

    for file in results_files:
        
        with open(f"{results_path}/{file}", 'r') as f:
            data = f.read()

        layer = file.split("_")[0]
        feature_id = file.split("_")[1].split(".")[0]

        data = data.split("──┘")[1]

        max_activation, res = get_data(data)

        results[layer][feature_id] = {
            "max_activation": max_activation,
            "results": res,
            "examples": f"../results/{layer}_top_examples.json"
        }

    return results

def highlight_example(feature_id, file_path, k):

    # read json
    with open(file_path, 'r') as f:
        data = json.load(f)

    examples = data[feature_id][:k]

    html_content = f'<h2>Top Activating Examples</h2>'
    count = 0
    
    for example in examples:
        if count >= k:
            break
        formatted_example = example.replace("[HIGHLIGHT]", "<mark>").replace("[/HIGHLIGHT]", "</mark>")
        html_content += f'<p>{formatted_example}</p>'
        count += 1
    
    return html_content

def generate_html():
    html_content = '<html><head><title>Neuron Feature Visualization</title></head><body>'
    html_content += '<h1>Neuron Feature Visualization</h1>'


    data = get_results("replicate")

    for layer, features in data.items():
        html_content += f"<h1>Layer {layer}</h1>"

        for feature_id, info in features.items():
            max_activation = info["max_activation"]
            results = info["results"]

            html_content += f"<h2>Feature {feature_id}</h2>"

            html_content += f"<p><strong>Max Activation:</strong> {max_activation}</p>"
            html_content += '<ul>'
            for i, exp in results.items():
                html_content += '<li>'
                html_content += explanation.format(i=i, d_score=exp["d_score"], gen_score=exp["gen_score"], text=exp["text"])
                html_content += '</li>'

            html_content += '</ul>'
            html_content += highlight_example(feature_id, info["examples"], 5)

    html_content += '</body></html>'

    with open('neuron_visualization.html', 'w') as file:
        file.write(html_content)


generate_html()