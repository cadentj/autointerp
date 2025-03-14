# %%
import os
import json
from collections import defaultdict

manual_path = "/share/u/caden/autointerp/experiments/manual"
html_path = os.path.join(manual_path, "html")

# Create html directory if it doesn't exist
os.makedirs(html_path, exist_ok=True)

data = defaultdict(dict)

def format_prompt(prompt):
    prompt = prompt.replace("Latent explanation:", "<b style='color: blue'>Latent explanation:</b><br>")
    for i in range(5):
        idx = i + 1
        prompt = prompt.replace(f"Example {idx}:", f"<b>Example {idx}:</b><br>")

    prompt = prompt.replace("\n", "<br>")

    return prompt

# Get all feature directories
feature_dirs = [d for d in os.listdir(manual_path) if os.path.isdir(f"{manual_path}/{d}") and d != "html"]

for feature_dir in feature_dirs:
    for prompt_file in os.listdir(f"{manual_path}/{feature_dir}"):
        with open(f"{manual_path}/{feature_dir}/{prompt_file}", "r") as f:
            results = json.load(f)
            prompt = results["prompt"][-1]['content']
            truth = results["truth"]
            quantiles = results["quantiles"]

            data[feature_dir][prompt_file] = {
                "prompt": format_prompt(prompt),
                "truth": truth,
                "quantiles": quantiles,
            }

# Create a separate HTML file for each feature
for i, (feature, feature_data) in enumerate(data.items()):
    prev_feature = feature_dirs[i-1] if i > 0 else None
    next_feature = feature_dirs[i+1] if i < len(feature_dirs)-1 else None
    
    feature_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{feature}</title>
        <style>
            body {{ 
                font-family: Arial, sans-serif; 
                margin: 20px;
                padding-bottom: 80px; /* Add padding to prevent content from being hidden behind nav buttons */
            }}
            .nav-buttons {{ 
                position: fixed;
                bottom: 20px;
                left: 50%;
                transform: translateX(-50%);
                display: flex;
                gap: 10px;
                background-color: rgba(255, 255, 255, 0.9); /* Add semi-transparent background */
                padding: 10px;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1); /* Add subtle shadow */
            }}
            .nav-button {{
                padding: 10px 20px;
                background-color: #007bff;
                color: white;
                text-decoration: none;
                border-radius: 5px;
            }}
            .nav-button:hover {{
                background-color: #0056b3;
            }}
            details {{
                margin: 10px 0;
                padding: 10px;
                background-color: #f8f9fa;
                border-radius: 5px;
            }}
            summary {{
                cursor: pointer;
                padding: 5px;
                font-weight: bold;
            }}
            summary:hover {{
                background-color: #e9ecef;
            }}
            .answer-section {{
                margin: 20px 0;
                padding: 10px;
                border-top: 1px solid #dee2e6;
            }}
            .answer-button {{
                padding: 10px 20px;
                background-color: #28a745;
                color: white;
                border: none;
                border-radius: 5px;
                cursor: pointer;
            }}
            .answer-button:hover {{
                background-color: #218838;
            }}
            .answer-content {{
                display: none;
                margin-top: 10px;
                padding: 10px;
                background-color: #e9ecef;
                border-radius: 5px;
            }}
        </style>
        <script>
            function toggleAnswer(promptFile) {{
                const answerContent = document.getElementById(`answer-${{promptFile}}`);
                const button = document.getElementById(`button-${{promptFile}}`);
                if (answerContent.style.display === 'none') {{
                    answerContent.style.display = 'block';
                    button.textContent = 'Hide Answer';
                }} else {{
                    answerContent.style.display = 'none';
                    button.textContent = 'Reveal Answer';
                }}
            }}
        </script>
    </head>
    <body>
        <div class="nav-buttons">
            {f'<a class="nav-button" href="{prev_feature}.html">Previous ({prev_feature})</a>' if prev_feature else ''}
            <a class="nav-button" href="index.html">Home</a>
            {f'<a class="nav-button" href="{next_feature}.html">Next ({next_feature})</a>' if next_feature else ''}
        </div>
        <h1>{feature}</h1>
    """

    def _map_quantiles(quantiles):
        updated = []
        for q in quantiles:
            if q == -1:
                updated.append("random")
            elif q == 0:
                updated.append("similar")
            else:
                updated.append(q)
        return updated

    def _map_truth(truth):
        updated = []
        for t in truth:
            if t:
                updated.append("activating")
            else:
                updated.append("non-activating")
        return updated

    for prompt_file, results in feature_data.items():
        safe_id = prompt_file.replace('.', '_')
        feature_html += f"""
        <details>
            <summary>{prompt_file}</summary>
            <pre>{results['prompt']}</pre>
            <div class="answer-section">
                <button id="button-{safe_id}" class="answer-button" onclick="toggleAnswer('{safe_id}')">Reveal Answer</button>
                <div id="answer-{safe_id}" class="answer-content">
                    <pre>{_map_truth(results['truth'])}</pre>
                    <pre>{_map_quantiles(results['quantiles'])}</pre>
                </div>
            </div>
        </details>
        """
    
    feature_html += """
    </body>
    </html>
    """
    
    with open(os.path.join(html_path, f"{feature}.html"), "w") as f:
        f.write(feature_html)

# Create the main index.html
index_html = """
<!DOCTYPE html>
<html>
<head>
    <title>Manual Experiments Index</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .feature-list {
            list-style: none;
            padding: 0;
        }
        .feature-link {
            display: block;
            padding: 10px;
            margin: 5px 0;
            background-color: #f8f9fa;
            text-decoration: none;
            color: #007bff;
            border-radius: 5px;
        }
        .feature-link:hover {
            background-color: #e9ecef;
        }
    </style>
</head>
<body>
    <h1>Manual Experiments</h1>
    <ul class="feature-list">
"""

for feature in feature_dirs:
    index_html += f'<li><a class="feature-link" href="{feature}.html">{feature}</a></li>'

index_html += """
    </ul>
</body>
</html>
"""

with open(os.path.join(html_path, "index.html"), "w") as f:
    f.write(index_html)

            
# %%
