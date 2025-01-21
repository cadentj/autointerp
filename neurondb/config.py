import yaml
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(CURRENT_DIR, "config.yaml"), "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    config["token"] = "sk-np-gpbPGO04zb8EUBH0XHY3hwoNcds0SjSBx7LwWOYaXXA0"

