import os
import yaml
import openai

PATH = os.path.dirname(os.path.abspath(__file__))

from .pydantics import ConfigModel

with open(os.path.join(PATH, "config.yaml"), "r") as file:
    CONFIG = ConfigModel(**yaml.safe_load(file))

if CONFIG is openai:
    CLIENT = openai.Client(CONFIG.API.APIKEY)