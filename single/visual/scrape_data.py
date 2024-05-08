# %%
import os
import re

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

provider = "replicate"
results_path = f"../results/{provider}"

# get list of all files in the results directory
results_files = os.listdir(results_path)

for file in results_files:
    
    with open(f"{results_path}/{file}", 'r') as f:
        data = f.read()

    data = data.split("──┘")[1]

    max_activation, explanations = get_data(data)

# %%

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

def highlight_example(examples, k):

    


