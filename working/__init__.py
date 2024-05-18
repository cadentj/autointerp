import os
import yaml

PATH = os.path.dirname(os.path.abspath(__file__))

from .pydantics import ConfigModel

with open(os.path.join(PATH, "config.yaml"), "r") as file:
    CONFIG = ConfigModel(**yaml.safe_load(file))