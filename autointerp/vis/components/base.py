import ipywidgets as widgets


class Component:
    def __init__(self, root: widgets.Output) -> None:
        self.root = root

    def __enter__(self):
        return self.root.__enter__()

    def __exit__(self, exc_type, exc_value, traceback):
        return self.root.__exit__(exc_type, exc_value, traceback)

    def clear(self):
        self.root.clear_output()
