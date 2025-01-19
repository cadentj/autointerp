import yaml
import os

with open("config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    config["token"] = os.environ["NEURONPEDIA_API_KEY"]

