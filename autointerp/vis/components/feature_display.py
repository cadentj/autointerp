from typing import List

import ipywidgets as widgets
from IPython.display import display, clear_output, HTML

from ...base import Example
from ..backend import InferenceResult

class FeatureDisplay:
    def __init__(self):
        self.feature_display = widgets.Output(
            layout=widgets.Layout(
                width="100%",
                height="300px",
                border="1px solid #ddd",
                padding="10px",
            )
        )

        self.root = self.feature_display

    def display(self, query_results: List[InferenceResult]):
        """Display the top features for the selected tokens."""
        with self.feature_display:
            clear_output()
            
            # Add CSS for tooltips
            tooltip_css = """
            <style>
                .token-wrapper {
                    position: relative;
                    display: inline-block;
                }
                .token-wrapper .tooltip {
                    visibility: hidden;
                    background-color: #555;
                    color: white;
                    text-align: center;
                    padding: 4px 8px;
                    border-radius: 4px;
                    position: absolute;
                    z-index: 1;
                    bottom: 125%;
                    left: 50%;
                    transform: translateX(-50%);
                    font-size: 12px;
                    white-space: nowrap;
                }
                .token-wrapper:hover .tooltip {
                    visibility: visible;
                }
            </style>
            """
            display(HTML(tooltip_css))
            display(HTML("<h3>Top Features</h3>"))

            for query_result in query_results:
                index = query_result.feature.index
                display(HTML(f"<h4>Feature {index}</h4>"))
                example_html = self._example_to_html(
                    query_result.inference_example
                )
                display(HTML(example_html))

    def _example_to_html(
        self,
        example: Example,
        threshold: float = 0.0,
    ) -> str:
        activations = example.activations
        str_tokens = example.str_tokens

        result = []
        max_act = activations.max()
        _threshold = max_act * threshold

        for i in range(len(str_tokens)):
            activation_str = f"Activation: {activations[i]:.3f}"
            if activations[i] > _threshold:
                opacity = 0.2 + 0.8 * (activations[i] / max_act)
                result.append(
                    f'<div class="token-wrapper"><mark style="opacity: {opacity:.2f}">{str_tokens[i]}'
                    f'</mark><span class="tooltip">{activation_str}</span></div>'
                )
            else:
                result.append(
                    f'<div class="token-wrapper"><span>{str_tokens[i]}'
                    f'</span><span class="tooltip">{activation_str}</span></div>'
                )

        return "".join(result)