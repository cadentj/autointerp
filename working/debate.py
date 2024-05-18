from .base import Environment

from .debater import Debater
from typing import List

class Debate(Environment):
    
    def __init__(
        self,
        debaters: List[Debater],
    ):
        super().__init__(debaters)

    def run(self):
        pass