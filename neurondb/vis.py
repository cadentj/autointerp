from typing import List, Tuple
from IPython.display import display
from circuitsvis.tokens import colored_tokens
import numpy as np

def normalize_activations(activations: List[float], max_activation: float) -> List[float]:
    """Normalize activation values by dividing by max activation"""
    # if not activations or max_activation == 0:
    #     return activations
    return [float(abs(x)) / max_activation for x in activations]

def show_neuron(tokens: List[str], activations: List[float], max_examples: int = 5) -> None:
    """Display neuron activations with colored tokens in a Jupyter notebook.
    
    Args:
        tokens: List of tokens or token strings
        activations: List of activation values
        max_examples: Maximum number of examples to show
    """
    try:
        for token, activation in zip(tokens[:max_examples], activations[:max_examples]):
            display(colored_tokens(token, activation))
    except ImportError:
        print("IPython is required for HTML display. Please install it or run this in a Jupyter notebook.")
        return

def export_neurons(
    neurons_data: List[Tuple[str, int, List[str], List[float], float, str]], 
    output_path: str = "vis.html"
) -> None:
    """Export neuron activations to an HTML file with highlighted tokens.
    
    Args:
        neurons_data: List of tuples containing (layer_id, neuron_index, (top_tokens, mid_tokens), (top_acts, mid_acts), max_activation, pos_str)
        output_path: Path to save the HTML file
    """
    print(f"Received {len(neurons_data)} neurons to visualize")
    
    for layer_id, index, tokens, activations, max_activation, pos_str in neurons_data:
        top_tokens, mid_tokens = tokens
        top_acts, mid_acts = activations
        print(f"Neuron {layer_id}-{index}:")
        print(f"  Top tokens: {len(top_tokens) if top_tokens else 0} examples")
        print(f"  Mid tokens: {len(mid_tokens) if mid_tokens else 0} examples")
        print(f"  Max activation: {max_activation}")

    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body { 
                font-family: Arial, sans-serif; 
                margin: 0;
                padding: 10px;
                line-height: 1.3;
                background-color: white;
                color: #333;
            }
            .container {
                width: 70%;
                margin: 0 auto;
                max-width: 1200px;
                background-color: white;
                padding: 20px;
                border-radius: 8px;
            }
            .neuron-section { 
                margin-bottom: 15px; 
                border-bottom: 1px solid #eee; 
                padding-bottom: 10px; 
            }
            .activation-section {
                margin: 8px 0;
            }
            .section-label {
                color: #888;
                font-size: 0.8em;
                margin: 3px 0;
            }
            .example { 
                margin: 5px 0; 
                padding: 5px; 
                background: #f8f8f8; 
                border-radius: 4px;
            }
            .token { 
                display: inline-block; 
                padding: 0 2px; 
                position: relative;
                transition: all 0.2s ease;
            }
            .token:hover {
                background-color: rgba(0, 0, 0, 0.1) !important;
                cursor: pointer;
            }
            .token:hover::after {
                content: attr(data-activation);
                position: absolute;
                bottom: 100%;
                left: 50%;
                transform: translateX(-50%);
                background: rgba(0, 0, 0, 0.8);
                color: white;
                padding: 4px 8px;
                border-radius: 4px;
                font-size: 12px;
                white-space: nowrap;
                z-index: 1;
            }
            h3 { color: #333; margin: 0 0 5px 0; font-size: 1em; }
            .pos-str { color: #666; font-size: 0.9em; margin: 0 0 8px 0; }
        </style>
    </head>
    <body>
        <div class="container">
    """

    for layer_id, index, tokens, activations, max_activation, pos_str in neurons_data:
        top_tokens, mid_tokens = tokens
        top_acts, mid_acts = activations

        html_content += f"<div class='neuron-section'>"
        html_content += f"<h3>{layer_id} - {index}</h3>"
        html_content += f"<div class='pos-str'><b>Max Activation: </b>{max_activation}</div>"
        if pos_str:
            html_content += f"<div class='pos-str'><b>Top Logits: </b>{pos_str}</div>"
        
        # Top activations section
        if top_tokens is not None:
            html_content += "<div class='activation-section'>"
            html_content += "<div class='section-label'>Top Activations</div>"
            if isinstance(top_tokens[0], list):
                for token_list, act_list in zip(top_tokens, top_acts):
                    html_content += "<div class='example'>"
                    normalized_acts = normalize_activations(act_list, max_activation)
                    for token, activation, norm_act in zip(token_list, act_list, normalized_acts):
                        color = "rgba(255, 150, 150, {})".format(norm_act) if float(activation) > 0 else "rgba(150, 150, 255, {})".format(norm_act)
                        html_content += f"<span class='token' data-activation='{activation:.3f}' style='background-color: {color}'>{token}</span>"
                    html_content += "</div>"
            else:
                for token_str, act_list in zip(top_tokens, top_acts):
                    html_content += "<div class='example'>"
                    tokens_split = token_str.split()
                    normalized_acts = normalize_activations(act_list, max_activation)
                    for token, activation, norm_act in zip(tokens_split, act_list, normalized_acts):
                        color = "rgba(255, 150, 150, {})".format(norm_act) if float(activation) > 0 else "rgba(150, 150, 255, {})".format(norm_act)
                        html_content += f"<span class='token' data-activation='{activation:.3f}' style='background-color: {color}'>{token}</span>"
                    html_content += "</div>"
            html_content += "</div>"

        # Middle activations section
        if mid_tokens is not None:
            html_content += "<div class='activation-section'>"
            html_content += "<div class='section-label'>Middle Activations</div>"
            if isinstance(mid_tokens[0], list):
                for token_list, act_list in zip(mid_tokens, mid_acts):
                    html_content += "<div class='example'>"
                    normalized_acts = normalize_activations(act_list, max_activation)
                    for token, activation, norm_act in zip(token_list, act_list, normalized_acts):
                        color = "rgba(255, 150, 150, {})".format(norm_act) if float(activation) > 0 else "rgba(150, 150, 255, {})".format(norm_act)
                        html_content += f"<span class='token' data-activation='{activation:.3f}' style='background-color: {color}'>{token}</span>"
                    html_content += "</div>"
            else:
                for token_str, act_list in zip(mid_tokens, mid_acts):
                    html_content += "<div class='example'>"
                    tokens_split = token_str.split()
                    normalized_acts = normalize_activations(act_list, max_activation)
                    for token, activation, norm_act in zip(tokens_split, act_list, normalized_acts):
                        color = "rgba(255, 150, 150, {})".format(norm_act) if float(activation) > 0 else "rgba(150, 150, 255, {})".format(norm_act)
                        html_content += f"<span class='token' data-activation='{activation:.3f}' style='background-color: {color}'>{token}</span>"
                    html_content += "</div>"
            html_content += "</div>"
        
        html_content += "</div>"

    html_content += "</div></body></html>"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"Visualization exported to {output_path}")

    return html_content

def wrap_html_for_notebook(html_content: str, height: int = 800) -> str:
    """Wrap HTML content in a scrollable div suitable for Jupyter notebook display.
    
    Args:
        html_content: The HTML content to wrap
        height: Height of the scrollable container in pixels
    
    Returns:
        str: The wrapped HTML content
    """
    return f"""
    <div style="height: {height}px; overflow-y: auto; border: 1px solid #ccc; padding: 10px;">
        {html_content}
    </div>
    """

def display_neurons(html_content: str, height: int = 800) -> None:
    """Display neurons visualization in a Jupyter notebook with scrollable container.
    
    Args:
        html_content: The HTML content to display
        height: Height of the scrollable container in pixels
    """
    from IPython.display import HTML, display
    wrapped_html = wrap_html_for_notebook(html_content, height)
    display(HTML(wrapped_html))