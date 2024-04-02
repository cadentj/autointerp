from typing import List

import torch

from .prompts import EXPLAIN_PROMPT
from .utils import gen_update, cached_gen

class Explainer:

    def __init__(
            self, 
            model, 
            mem: List
        ):
        self.model = model

        self.mem = mem

    def __call__(
            self, 
        ) -> None:
        """Run the evaluator on some feature activating phrases for a given neuron.

        Args:
            actions (List[str]): List of phrases
        
        Returns:
            None
        """
        explain = {
            "role" : "user",
            "content" : EXPLAIN_PROMPT
        }

        self.mem[-1].agent.append(explain)
        
        return gen_update(self)