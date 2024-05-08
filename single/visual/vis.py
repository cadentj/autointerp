#%%
import os
import re
import json
from collections import defaultdict

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
    with open(file_path, 'r') as f:
        data = json.load(f)

    examples = data[feature_id][:k]
    html_content = '<h3>Top Activating Examples</h3>'
    
    for example in examples:
        formatted_example = example.replace("[HIGHLIGHT]", "<mark>").replace("[/HIGHLIGHT]", "</mark>")
        html_content += f'<p>{formatted_example}</p>'
    
    return html_content

def neuronpedia_link(layer, feature_id):
    return f"https://www.neuronpedia.org/gpt2-small/{layer}-res-jb/{feature_id}"

def github_link(provider, layer, feature_id):
    return f"https://github.com/cadentj/autointerp/blob/main/single/results/{provider}/{layer}_{feature_id}.txt"

def generate_html():
    html_content = '<html><head><title>Neuron Feature Visualization</title>'
    html_content += '<style>a { color: black; }</style>'  
    html_content += '</head><body>'
    html_content += '<h1>Neuron Feature Visualization</h1>'

    data_replicate = get_results("replicate")
    data_openai = get_results("openai")

    for layer, features in data_replicate.items():
        html_content += f"<h1>Layer {layer}</h1>"

        for feature_id, info_rep in features.items():
            info_openai = data_openai[layer].get(feature_id, {'results': {}})

            max_activation = info_rep["max_activation"]
            results_rep = info_rep["results"]
            results_openai = info_openai["results"]

            link = neuronpedia_link(layer, feature_id)
            html_content += f"<h2><a href='{link}'>Feature {feature_id}</a></h2>"

            html_content += highlight_example(feature_id, info_rep["examples"], 15)

            html_content += f"<p><strong>Max Activation:</strong> {max_activation}</p>"

            max_explanations = max(len(results_rep), len(results_openai))

            for i in range(max_explanations):
                exp_rep = results_rep.get(i, None)
                exp_openai = results_openai.get(i, None)

                html_content += f'<div style="display:flex;">'

                if exp_rep:
                    link = github_link("replicate", layer, feature_id)
                    html_content += f'<div style="flex:1; padding:10px;">'
                    html_content += f"<h3><a href='{link}'>Replicate Explanation {i}</a></h3>"
                    html_content += f"<p><b>Detection Score:</b> {exp_rep['d_score']}<br><b>Generation Score:</b> {exp_rep['gen_score']}</p>"
                    html_content += f"<p>{exp_rep['text']}</p>"
                    html_content += '</div>'
                else:
                    html_content += '<div style="flex:1; padding:10px;"></div>'

                if exp_openai:
                    link = github_link("openai", layer, feature_id)
                    html_content += f'<div style="flex:1; padding:10px;">'
                    html_content += f"<h3><a href='{link}'>OpenAI Explanation {i}</a></h3>"
                    html_content += f"<p><b>Detection Score:</b> {exp_openai['d_score']}<br><b>Generation Score:</b> {exp_openai['gen_score']}</p>"
                    html_content += f"<p>{exp_openai['text']}</p>"
                    html_content += '</div>'
                else:
                    html_content += '<div style="flex:1; padding:10px;"></div>'

                html_content += '</div>'

    html_content += '</body></html>'

    with open('neuron_visualization.html', 'w') as file:
        file.write(html_content)

generate_html()

# %%
