from typing import List, Dict, Any

import ipywidgets as widgets
from IPython.display import display, clear_output, HTML

from .backend import Backend, FeatureFn, sample_feature_extraction

BUTTON_STYLE = """
<style>
.{token_box_class} .jupyter-button {{
    width: auto !important;
    height: auto !important;
}}
</style>
"""

def make_dashboard(cache_dir: str, feature_fn: FeatureFn):
    backend = Backend(cache_dir, feature_fn)
    return FeatureVisualizationDashboard(backend).display()


class FeatureVisualizationDashboard:
    """Dashboard for visualizing features in a neural network."""

    def __init__(self, model: Backend):
        """Initialize the dashboard components."""
        self.model = model

        # Input components
        self.text_input = widgets.Textarea(
            placeholder="Enter text to analyze...",
            layout=widgets.Layout(width="100%", height="100px"),
        )

        self.tokenize_button = widgets.Button(
            description="Tokenize",
            button_style="primary",
            layout=widgets.Layout(width="auto"),
        )

        # Token display components
        self.token_display = widgets.Output(
            layout=widgets.Layout(
                width="100%", border="1px solid #ddd", padding="10px"
            )
        )

        self.token_widgets = []
        self.selected_tokens = set()

        # Feature analysis components
        self.run_button = widgets.Button(
            description="Run",
            button_style="success",
            layout=widgets.Layout(width="auto"),
            disabled=True,
        )

        self.feature_display = widgets.Output(
            layout=widgets.Layout(
                width="100%",
                height="300px",
                border="1px solid #ddd",
                padding="10px",
            )
        )

        # Top level container
        self.input_container = widgets.VBox(
            [
                widgets.Label("Enter text to analyze:"),
                self.text_input,
                self.tokenize_button,
            ]
        )

        self.analysis_container = widgets.VBox(
            [
                widgets.Label("Select tokens and analyze:"),
                self.token_display,
                self.run_button,
                self.feature_display,
            ]
        )

        self.main_container = widgets.VBox(
            [self.input_container, self.analysis_container]
        )

        # Wire up event handlers
        self.tokenize_button.on_click(self._on_tokenize_clicked)
        self.run_button.on_click(self._on_run_clicked)

        self.run_callback = sample_feature_extraction

    def _on_tokenize_clicked(self, b):
        """Handle tokenize button click."""

        text = self.text_input.value
        if not text.strip():
            with self.token_display:
                clear_output()
                print("Please enter some text first")
            return

        tokens = self.model.tokenize(text, to_str=True)
        self._display_tokens(tokens)
        self.run_button.disabled = False

        # Hide text input, show tokens instead
        self.input_container.children = [
            widgets.Label("Tokenized text:"),
            self.token_display,
        ]

    def _display_tokens(self, tokens: List[str]):
        """Display tokenized text with selectable boxes resembling spans."""
        self.token_widgets = []
        self.selected_tokens = set()

        with self.token_display:  # Use the output widget context
            clear_output(wait=True)  # Clear previous content

            # Unique class for this instance's token container
            token_box_class = f"token-box-{id(self)}"
            token_box = widgets.HBox(
                layout=widgets.Layout(flex_wrap="wrap", padding="5px")
            )  # Add padding to container
            token_box.add_class(token_box_class)

            # Define colors for selection states
            color_unselected = "transparent"
            color_selected = "lightblue"

            # Generate CSS to override default button styles
            # Targeting buttons specifically within our unique container class
            # Display the CSS within the Output widget
            display(HTML(BUTTON_STYLE.format(token_box_class=token_box_class)))

            # Create the buttons (now styled by the CSS above)
            for i, token in enumerate(tokens):
                token_button = widgets.Button(
                    description=token,
                    # Layout settings might be redundant but can help structure
                    layout=widgets.Layout(margin="0px 1px", padding="1px 0px"),
                    style={
                        "button_color": color_unselected
                    },  # Set initial background color
                )

                # Click handler remains the same, toggling the background color
                def create_selection_handler(idx, btn):
                    def handler(b):
                        if idx in self.selected_tokens:
                            self.selected_tokens.remove(idx)
                            btn.style.button_color = color_unselected
                        else:
                            self.selected_tokens.add(idx)
                            btn.style.button_color = color_selected

                    return handler

                token_button.on_click(create_selection_handler(i, token_button))
                self.token_widgets.append(token_button)

            token_box.children = self.token_widgets  # Assign buttons to HBox
            # Display the token box within the Output widget (after the CSS)
            display(token_box)

    def _on_run_clicked(self, b):
        """Handle run button click."""
        if not self.selected_tokens:
            with self.feature_display:
                clear_output()
                print("Please select at least one token")
            return

        # Convert set to sorted list to ensure consistent ordering
        selected_indices = sorted(list(self.selected_tokens))

        with self.feature_display:
            clear_output()
            print(f"Analyzing features for selected tokens: {selected_indices}")

        features = self.model.inference_query(
            self.text_input.value,
            selected_indices,
        )
        self._display_features(features)

    def _display_features(self, features: Dict[str, Any]):
        """Display the top features for the selected tokens."""
        with self.feature_display:
            clear_output()
            # The actual display will depend on the format of the features returned
            # This is a placeholder for future implementation
            display(HTML("<h3>Top Features</h3>"))

            # Example display, adjust based on actual feature data format

            for token_idx, query_result in features.items():
                display(HTML(f"<h4>Feature {token_idx}</h4>"))
                display(HTML(f"Max activation: {query_result.max_activation}"))
                display(HTML(query_result.context_activation))
                cached_html = "<br><br>".join(query_result.cached_activations)
                display(HTML(cached_html))


    def display(self):
        """Display the dashboard."""
        display(self.main_container)
