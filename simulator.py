from typing import List
from torch import Tensor
from dataclasses import dataclass


@dataclass

class Feature:
    tokens: List[str]
    acts: Tensor
    n_acts: Tensor = None

class Agent:

    def __init__(self, model):
        self.model = model

        self.prompt_builder = PromptBuilder(model.tokenizer)

        self.memory = []
        self.scores = []

    def explain(self, feature: Feature):
        """Generate a detailed explaination of feature activations

        Args:
            feature (Feature): The feature to explain, unnormalized activations

        Returns:
            explaination (str): A detailed explaination of the feature activations
        """
        
        prompt = ""

        with self.model.generate(prompt, max_new_tokens=100):
            tokens = self.model.generator.output.save() 

        decoded_tokens = self.model.tokenizer.decode(tokens)

    def action(self):
        """Given some explaination, generate phrases which maximize activations

        Args: 
            explaination (str): A detailed explaination of the feature activations

        Returns:
            phrases (List[str]): Phrases which maximize activations
        """
        pass


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

class PromptBuilder:

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def build_explaination(self):
        pass

    def build_action(self):
        pass

    def build_reflection(self):
        pass

class RewardEngine:

    def __init__(self, model, dictionaries):
        self.model = model
        self.dictionaries = dictionaries

    def score(self):
        pass

