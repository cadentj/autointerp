from abc import ABC, abstractmethod

from .agent import Agent
from typing import List 

class Environment(ABC):

    def __init__(
        self, 
        agents: List[Agent]
    ):
        self.agents = agents

    @abstractmethod
    def run():
        pass