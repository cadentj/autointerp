from typing import List

from .prompts import REFLECTION_PROMPT
from .utils import gen_update

class SelfReflector:

    def __init__(
            self, 
            model, 
            mem: List
        ):
        self.model = model
        self.mem = mem

    def __call__(self) -> bool:
        """Generate a plan for what went wrong/right on a trial.

        Args:
            None

        Returns:
            bool: True if the trial was successful, False if not
        """

        if self.check_success():
            return True
        else:
            reflection = {
                "role" : "user",
                "content" : REFLECTION_PROMPT.format(
                    results = self.list_scores()
                )
            }

            self.mem[-1].agent.append(reflection)

            self.mem[-1].self_reflector = [
                reflection,
                {
                    "role" : "assistant",
                    "content" : gen_update(self)
                }
            ]

            return False

    def check_success(self) -> bool:
        """Check if the trial was successful.

        Args:
            None

        Returns:
            bool: True if the trial was successful, False if not
        """

        scores = list(self.mem[-1].evaluator.values())
        avg_score = sum(scores) / len(scores)

        if avg_score > 7:
            return True
    
        return False

    def list_scores(self) -> str:
        """Generate a string representation of phrases and scores. 

        Args:
            None
        
        Returns:
            str: String representation of phrases and scores
        """
        scores = self.mem[-1].evaluator
        
        flattened = ""
        for k, v, in scores.items():
            flattened += f"{k}: {v}\n"

        return flattened