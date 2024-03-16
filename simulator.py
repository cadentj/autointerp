from typing import List
from torch import Tensor
from dataclasses import dataclass

SYSTEM_PROMPT = ""
ACTION_PROMPT = ""

@dataclass
class Feature:
    tokens: List[str]
    acts: Tensor
    n_acts: Tensor = None

def gen(model, messages, remote=False):
    with model.generate(messages, max_new_tokens=100, remote=remote):
        tokens = model.generator.output.save()

    return model.tokenizer.decode(tokens)

class Agent:

    def __init__(self, model, reward_engine):
        self.model = model
        self.tokenizer = model.tokenizer

        self.reward_engine = reward_engine

        self.scores = []
        self.memory = None
        self.trials = {}

    def explain(self, feature: Feature):
        """Generate a detailed explaination of feature activations

        Args:
            feature (Feature): The feature to explain, unnormalized activations

        Returns:
            explaination (str): A detailed explaination of the feature activations
        """
        
        if not self.memory:
            
            system = {
                "role" : "system",
                "content" : SYSTEM_PROMPT
            }

            environment = {
                "role" : "user",
                "content" : self.generate_environment(feature)
            }

            messages = [system, environment]
        else: 
            messages = self.memory
            # TODO: no idea

        return gen(self.model, messages)

    def generate_environment(self, feature: Feature):
        pass


    def action(self, messages: str):
        """Given some explaination, generate phrases which maximize activations

        Args: 
            messages (str): Message history with the most recent being an explaination

        Returns:
            phrases (List[str]): Phrases which maximize activations
        """
        
        messages.append({
            "role" : "user",
            "content" : ACTION_PROMPT
        })

        return gen(self.model, messages)

    def parse_action(self):
        """Given some phrases, parse them into scorable tokens for the RewardEngine

        Args:
            phrases (List[str]): Phrases which maximize activations

        Returns:
            tokens (List[str]): Scorable tokens
        """
        pass

    def reward(self):
        """Call the RewardEngine to score the tokens
        
        Args:
            tokens (List[str]): Scorable tokens

        Returns:
            score (float): The score of the tokens
        """
        pass

    def reflect(self):
        """Reflect on the score and update long term memory

        Args:
            score (float): The score of the tokens
        
        Returns:
            None
        """
        pass

    def __call__(self, feature: Feature):

        explaination = self.explain(feature)

        action = self.action(explaination)
        parsed_action = self.parse_action(action)

        reward = self.reward_engine.score(parsed_action)
        self.scores.append(reward)

        reflection = self.reflect(reward)

        self.trial[0] = reflection

class RewardEngine:

    def __init__(self, model, dictionaries):
        self.model = model
        self.dictionaries = dictionaries

    def score(self):
        pass

