from typing import List

import ipywidgets as widgets
from IPython.display import display, clear_output, HTML

from ...base import Example
from ..backend import InferenceResult
from .base import Component


TOOLTIP = """
<style>
.tooltip {
    position: relative;
    display: inline-block;
}

.tooltip .tooltiptext {
    visibility: hidden;
    width: 120px;
    background-color: black;
    color: #fff;
    text-align: center;
    border-radius: 6px;
    padding: 5px 0;

    /* Position the tooltip */
    position: absolute;
    z-index: 1000;
    top: 100%;
    left: 120%;
}

.tooltip:hover .tooltiptext {
    visibility: visible;
}

.activating-example {
    margin: 3px 0;
    padding: 3px;
    background-color: #f5f5f5;
}
</style>
"""

ACTIVATING_EXAMPLE_WRAPPER = """
<div class="activating-example">
    {example}
</div>
"""

HIGHLIGHTED_TOKEN_WRAPPER = """
<span class="tooltip">
    <span style="background-color: rgba(80, 211, 153, {opacity:.2f})">{token}</span>
    <span class="tooltiptext">{activation:.2f}</span>
</span>
"""


class FeatureDisplay(Component):
    def __init__(self):
        self.feature_display = widgets.Output(
            layout=widgets.Layout(
                width="100%",
                height="100%",
                border="1px solid #ddd",
                padding="10px",
            )
        )

        super().__init__(self.feature_display)

    def display(self, query_results: List[InferenceResult]):
        """Display the top features for the selected tokens."""
        with self.feature_display:
            clear_output()

            display(HTML(TOOLTIP))
            display(HTML("<h3>Top Features</h3>"))

            for i, query_result in enumerate(query_results):
                index = query_result.feature.index
                display(HTML(f"<h4>Feature {index}</h4>"))

                inference_html = self._example_to_html(
                    query_result.inference_example
                )
                display(
                    HTML(
                        ACTIVATING_EXAMPLE_WRAPPER.format(
                            example=inference_html
                        )
                    )
                )

                display(HTML("<hr>"))

                # Only add dropdown if there are activating examples
                if query_result.feature.activating_examples:
                    # Display activating examples directly
                    for example in query_result.feature.activating_examples:
                        example_html = self._example_to_html(example)
                        display(
                            HTML(
                                ACTIVATING_EXAMPLE_WRAPPER.format(
                                    example=example_html,
                                )
                            )
                        )

    def _example_to_html(
        self,
        example: Example,
        threshold: float = 0.0,
    ) -> str:
        str_tokens = example.str_tokens
        activations = example.activations

        result = []
        max_act = activations.max()
        _threshold = max_act * threshold

        for i in range(len(str_tokens)):
            if activations[i] > _threshold:
                # Calculate opacity based on activation value (normalized between 0.2 and 1.0)
                opacity = 0.2 + 0.8 * (activations[i] / max_act)
                result.append(
                    HIGHLIGHTED_TOKEN_WRAPPER.format(
                        token=str_tokens[i],
                        opacity=opacity,
                        activation=activations[i].item(),
                    )
                )
            else:
                result.append(str_tokens[i])
        return "".join(result)
