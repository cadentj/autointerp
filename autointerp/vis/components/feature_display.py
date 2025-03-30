from typing import List

import ipywidgets as widgets
from IPython.display import display, clear_output, HTML

from ...base import Example
from ..backend import InferenceResult

TOOLTIP_STYLES = """
<style>
/* CSS for custom instant tooltip */
.tooltip-span {
  position: relative; /* Needed for absolute positioning of the tooltip */
  display: inline-block;
  text-decoration: none; /* Ensure no underline */
  border-bottom: 1px dotted black; /* Optional: visual cue */
  cursor: help; /* Optional: change cursor on hover */
}

/* Tooltip pseudo-element */
.tooltip-span::after {
  content: attr(data-tooltip); /* Use the data-tooltip attribute content */
  position: absolute;
  background-color: #333; /* Tooltip background */
  color: #fff;           /* Tooltip text color */
  padding: 4px 8px;
  border-radius: 4px;
  font-size: 0.9em;
  white-space: nowrap; /* Prevent tooltip text wrapping */
  z-index: 10;          /* Ensure tooltip is on top */

  /* Positioning: above the element */
  left: 50%;
  bottom: 100%;
  transform: translateX(-50%);
  margin-bottom: 5px; /* Space between element and tooltip */

  /* Visibility: hidden by default, visible on hover */
  visibility: hidden;
  opacity: 0;
  /* Remove transition for instant appearance */
  /* transition: opacity 0.2s, visibility 0.2s; */
}

/* Show tooltip on hover */
.tooltip-span:hover::after {
  visibility: visible;
  opacity: 1;
}
</style>
"""


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

    def display(self, query_results: List[InferenceResult]):
        """Display the top features for the selected tokens."""
        with self.feature_display:
            clear_output()

            display(HTML(TOOLTIP_STYLES))
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
            activation_value = activations[i]  # Store activation value
            if activation_value > _threshold:
                # Calculate opacity based on activation value (normalized between 0.2 and 1.0)
                opacity = 0.2 + 0.8 * (activation_value / max_act)
                # Add background-color: red; wrap token in span with class and data-tooltip
                tooltip_text = f"Activation: {activation_value:.2f}"
                result.append(
                    f'<mark style="background-color: red; opacity: {opacity:.2f};">'
                    # Added class="tooltip-span" and data-tooltip attribute
                    f'<span class="tooltip-span" data-tooltip="{tooltip_text}">{str_tokens[i]}</span>'
                    f"</mark>"
                )
            else:
                result.append(str_tokens[i])

        return "".join(result)
