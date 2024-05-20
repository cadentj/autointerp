import threading
from typing import List
from functools import partial
from abc import ABC, abstractmethod

from ..agents import Debater
from ..history import History

class Debate(ABC):
    
    def __init__(
        self,
        debaters: List[Debater],
    ):        
        self.history = History()
        self.debaters = debaters

    @abstractmethod
    def run(
        self,
        max_rounds: int = 3,
    ):
        pass    
    
    def debate(self):
        threads = []

        for d in self.debaters:
            history_add = partial(self.history.add, d.id)

            thread = threading.Thread(
                target=d.execute,
                args=(
                    self.history.history[d.id], # this is turn
                    {}, 
                    lambda turn, add=history_add: add(turn)
                ),
            )
            
            threads.append(thread)
            thread.start()

        for t in threads:
            t.join()

    

