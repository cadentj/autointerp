from typing import List
from torch import Tensor

class Simulator:
    def __init__(self, model, data, target):
        self.model = model
        self.data = data
        self.target = target

    def simulate(self, instance):
        pass