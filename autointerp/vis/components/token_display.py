from typing import List

import ipywidgets as widgets
from IPython.display import display, clear_output, HTML


BUTTON_STYLE = """
<style>
.{token_box_class} .jupyter-button {{
    width: auto !important;
    white-space: pre-wrap !important;
    display: inline-block !important;
    padding: 0px !important;
    margin: 0px !important;
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
            # Use VBox to stack HBox rows for explicit line breaks
            # Or keep HBox and insert a line break widget. Let's try the latter first.
            token_container = widgets.HBox(
                layout=widgets.Layout(
                    flex_flow="row wrap",
                    padding="0px",
                    margin="0px",
                )
            )
            token_container.add_class(token_box_class)

            # Define colors for selection states
            color_unselected = "transparent"
            color_selected = "lightblue"

            # Generate CSS to override default button styles
            display(HTML(BUTTON_STYLE.format(token_box_class=token_box_class)))

            # Create the buttons (now styled by the CSS above)
            for i, token in enumerate(tokens):
                is_newline = token == "\n"
                display_token = "\\n" if is_newline else token

                token_button = widgets.Button(
                    description=display_token,
                    layout=widgets.Layout(
                        padding="0px",
                        margin="0px",
                        border="1px dashed lightgrey" if is_newline else "none",
                    ),
                    style={"button_color": color_unselected},
                    tooltip=f"Token {i}: '{token}'",  # Add tooltip for clarity
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

                # If it's a newline token, add a widget to force a break
                if is_newline:
                    # An empty HTML widget spanning the full width forces a wrap
                    line_break = widgets.HTML(
                        "<div style='width: 100%; height: 0;'></div>"
                    )
                    token_widgets.append(line_break)

            token_container.children = (
                token_widgets  # Assign all widgets to container
            )
            # Display the token container within the Output widget
            display(token_container)

    def clear(self):
        """Clear the token display."""
        self.token_display.clear_output()
