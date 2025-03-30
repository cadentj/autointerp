from typing import List

import ipywidgets as widgets
from IPython.display import display, clear_output, HTML


BUTTON_STYLE = """
<style>
.{token_box_class} .jupyter-button {{
    width: auto !important;
    height: auto !important;
}}
</style>
"""


class TokenDisplay:
    def __init__(self):
        self.token_display = widgets.Output(
            layout=widgets.Layout(
                width="100%", border="1px solid #ddd", padding="10px"
            )
        )

        self.selected_tokens = set()
        self.root = self.token_display

    def display(self, tokens: List[str]):
        """Display tokenized text with selectable boxes resembling spans."""
        token_widgets = []
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
                token_widgets.append(token_button)

            token_box.children = token_widgets  # Assign buttons to HBox
            # Display the token box within the Output widget (after the CSS)
            display(token_box)
