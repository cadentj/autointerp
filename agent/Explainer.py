from typing import List

import torch

from .prompts import EXPLAIN_PROMPT
from .utils import gen_update

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
        
        self.mem[-1].agent.append({
            "role" : "user",
            "content" : EXPLAIN_PROMPT
        })

        response = gen_update(self)

        self.mem[-1].agent.append({
            "role" : "assistant",
            "content" : response
        })
        
        return response