import ipywidgets as widgets
from IPython.display import display, clear_output

from .backend import Backend, FeatureFn
from .components.token_display import TokenDisplay
from .components.feature_display import FeatureDisplay


def make_dashboard(cache_dir: str, feature_fn: FeatureFn, in_memory: bool = False):
    backend = Backend(cache_dir, feature_fn, in_memory=in_memory)
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
        self.token_display = TokenDisplay()

        # Feature analysis components
        self.run_button = widgets.Button(
            description="Run",
            button_style="success",
            layout=widgets.Layout(width="auto"),
            disabled=True,
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

        self.analysis_container = widgets.VBox(
            [
                widgets.Label("Select tokens and analyze:"),
                self.token_display.root,
                self.run_button,
                self.feature_display.root,
            ]
        )

        self.main_container = widgets.VBox(
            [self.input_container, self.analysis_container]
        )

        # Wire up event handlers
        self.tokenize_button.on_click(self._on_tokenize_clicked)
        self.run_button.on_click(self._on_run_clicked)

    def _on_tokenize_clicked(self, b):
        """Handle tokenize button click."""

        text = self.text_input.value
        if not text.strip():
            with self.token_display.root:
                clear_output()
                print("Please enter some text first")
            return

        tokens = self.model.tokenize(text, to_str=True)
        self.token_display.display(tokens)
        self.run_button.disabled = False

        # Hide text input, show tokens instead
        self.input_container.children = [
            widgets.Label("Tokenized text:"),
            self.token_display.root,
        ]

    def _on_run_clicked(self, b):
        """Handle run button click."""
        selected_indices = sorted(list(self.token_display.selected_tokens))

        if not selected_indices:
            selected_indices = "all"

        with self.feature_display.root:
            clear_output()
            print(f"Analyzing features for selected tokens: {selected_indices}")

        query_results = self.model.inference_query(
            self.text_input.value,
            selected_indices,
        )
        self.feature_display.display(query_results)

    def display(self):
        """Display the dashboard."""
        display(self.main_container)

             