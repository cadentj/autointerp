from typing import List

import ipywidgets as widgets
from IPython.display import display, clear_output, HTML

from ...base import Example
from ..backend import InferenceResult


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
    <strong>Example {index}:</strong>
    {example}
</div>
"""

HIGHLIGHTED_TOKEN_WRAPPER = """
<span class="tooltip">
    <span style="background-color: rgba(0, 0, 255, {opacity:.2f})">{token}</span>
    <span class="tooltiptext">{activation:.2f}</span>
</span>
"""


class FeatureDisplay:
    def __init__(self):
        self.feature_display = widgets.Output(
            layout=widgets.Layout(
                width="100%",
                height="100%",
                border="1px solid #ddd",
                padding="10px",
            )
        )

        self.root = self.feature_display
        self.example_outputs = {}  # Store outputs for each example

    def display(self, query_results: List[InferenceResult]):
        """Display the top features for the selected tokens."""
        with self.feature_display:
            clear_output()
            self.example_outputs = {}  # Reset outputs

            display(HTML(TOOLTIP))
            display(HTML("<h3>Top Features</h3>"))

            for i, query_result in enumerate(query_results):
                feature_id = f"feature-{i}-{query_result.feature.index}"
                index = query_result.feature.index
                display(HTML(f"<h4>Feature {index}</h4>"))

                # Display inference example with special styling
                inference_html = self._example_to_html(
                    query_result.inference_example
                )
                display(inference_html)

                # Only add dropdown if there are activating examples
                if query_result.feature.activating_examples:
                    # Create an output widget to display the selected example
                    example_output = widgets.Output()
                    self.example_outputs[feature_id] = example_output

                    # Create dropdown widget
                    examples_count = len(
                        query_result.feature.activating_examples
                    )
                    dropdown = self._make_dropdown(
                        example_output, examples_count, query_result
                    )
                    display(dropdown)

    def _make_dropdown(
        self,
        example_output: widgets.Output,
        examples_count: int,
        query_result: InferenceResult,
    ):
        dropdown = widgets.Accordion(
            children=[example_output],
            selected_index=None,  # Start closed
            titles=(f"Show {examples_count} Activating Examples",),
        )

        # Register callback for when dropdown is opened/closed
        def on_selected_change(change):
            if change["new"] is not None:  # If opened
                with example_output:
                    clear_output()
                    # Display all activating examples
                    for j, example in enumerate(
                        query_result.feature.activating_examples
                    ):
                        example_html = self._example_to_html(example)
                        display(
                            HTML(
                                ACTIVATING_EXAMPLE_WRAPPER.format(
                                    index=j + 1,
                                    example=example_html,
                                )
                            )
                        )

        dropdown.observe(on_selected_change, names="selected_index")
        return dropdown

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

    def clear(self):
        self.example_outputs = {}
        self.feature_display.clear_output()
