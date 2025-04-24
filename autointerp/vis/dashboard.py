from typing import List

import ipywidgets as widgets
from IPython.display import display, clear_output

from .backend import Backend, FeatureFn
from .components.token_display import TokenDisplay
from .components.feature_display import FeatureDisplay


def make_dashboard(
    cache_dir: str, feature_fn: FeatureFn, in_memory: bool = False
):
    backend = Backend(cache_dir, feature_fn, in_memory=in_memory)
    return FeatureVisualizationDashboard(backend).display()


def make_feature_display(
    cache_dir: str, features: List[int], **load_kwargs
):
    backend = Backend(cache_dir, None, load_model=False)
    dash = FeatureDisplay()

    display(dash.root)

    with dash:
        loaded_features = backend.query(features, **load_kwargs)

    loaded_features = list(loaded_features.values())
    dash.display(loaded_features)


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
        self.token_display = TokenDisplay()

        # Feature analysis components
        self.run_button = widgets.Button(
            description="Run",
            button_style="success",
            layout=widgets.Layout(width="auto"),
            disabled=True,
        )
        self.reset_button = widgets.Button(
            description="Reset",
            button_style="warning",
            layout=widgets.Layout(width="auto"),
            disabled=True,
        )

        # Add k input widget
        self.k_input = widgets.IntText(
            value=10,  # Default value for k
            description="Top K:",
            style={
                "description_width": "initial"
            },  # Adjust style for description
            layout=widgets.Layout(width="120px"),  # Give it a fixed width
            disabled=False,  # K can be set anytime
        )

        self.feature_display = FeatureDisplay()

        # Top level container
        self.input_container = widgets.VBox(
            [
                widgets.Label("Enter text to analyze:"),
                self.text_input,
                self.tokenize_button,
            ]
        )

        # Group Run, Reset buttons, and K input
        # Place k_input before the buttons
        self.action_controls = widgets.HBox(
            [self.k_input, self.run_button, self.reset_button],
            layout=widgets.Layout(
                align_items="center"
            ),  # Align items vertically
        )

        self.analysis_container = widgets.VBox(
            [
                widgets.Label("Select tokens and analyze:"),
                self.token_display.root,
                self.action_controls,
                self.feature_display.root,
            ]
        )
        self.analysis_container.layout.display = "none"

        self.main_container = widgets.VBox(
            [self.input_container, self.analysis_container]
        )

        # Wire up event handlers
        self.tokenize_button.on_click(self._on_tokenize_clicked)
        self.run_button.on_click(self._on_run_clicked)
        self.reset_button.on_click(self._on_reset_clicked)

    def _on_tokenize_clicked(self, b):
        """Handle tokenize button click."""

        text = self.text_input.value
        if not text.strip():
            self.analysis_container.layout.display = "block"
            with self.feature_display:
                clear_output()
                print("Please enter some text first")

            self.token_display.clear()
            self.run_button.disabled = True
            self.reset_button.disabled = True
            return

        tokens = self.model.tokenize(text, to_str=True)
        self.token_display.display(tokens)

        self.run_button.disabled = False
        self.reset_button.disabled = False
        self.input_container.layout.display = "none"
        self.analysis_container.layout.display = "block"
        self.feature_display.clear()

    def _on_run_clicked(self, b):
        """Handle run button click."""
        selected_indices = sorted(list(self.token_display.selected_tokens))
        k_value = self.k_input.value  # Get k from the input widget

        assert k_value > 0 and k_value <= 100, "k must be between 1 and 50"

        if not selected_indices:
            selected_indices = "all"

        self.feature_display.clear()
        with self.feature_display:
            clear_output()
            print(
                f"Analyzing top {k_value} features,",
                f"for selected tokens: {selected_indices}",
            )

            # Put this in here so tqdm is displayed in the output widget
            query_results = self.model.inference_query(
                self.text_input.value,
                selected_indices,
                k=k_value,
            )
        self.feature_display.display(query_results)

    def _on_reset_clicked(self, b):
        """Handle reset button click."""
        self.token_display.clear()
        self.feature_display.clear()

        self.text_input.value = ""
        self.run_button.disabled = True
        self.reset_button.disabled = True
        # Reset k input to default value
        self.k_input.value = 10
        self.analysis_container.layout.display = "none"
        self.input_container.layout.display = "block"

    def display(self):
        """Display the dashboard."""
        display(self.main_container)
