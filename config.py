import yaml
import os

with open("config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    config["token"] = "sk-np-gpbPGO04zb8EUBH0XHY3hwoNcds0SjSBx7LwWOYaXXA0"

