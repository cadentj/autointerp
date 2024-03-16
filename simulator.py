from typing import List
from torch import Tensor
import torch
from dataclasses import dataclass
import re

SYSTEM_PROMPT = ""
ACTION_PROMPT = "Given your explaination, please output a phrase that maximizes activations. You should only output the phrase, and nothing else before or after it. Surround your phrase with brackets {YOUR_PHRASE} "
RE_EXPLAIN_PROMPT = "Given the self reflections..."
REFLECTION_PROMPT = "Given the score..."

@dataclass 
class Location:
    feature_type: str # mlp, resid, attn
    index: int # dictionary index
    layer: int # layer index

@dataclass
class Feature:
    tokens: List[str]
    acts: Tensor
    n_acts: Tensor = None
    location: Location = None

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
        self.trials = {}

    def explain(self, feature: Feature):
        """Generate a detailed explaination of feature activations

        Args:
            feature (Feature): The feature to explain, unnormalized activations

        Returns:
            explaination (str): A detailed explaination of the feature activations
        """
        
        if not self.scores:
            
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
            re_explain = {
                "role" : "system",
                "content" : RE_EXPLAIN_PROMPT
            }

            messages.append(re_explain)

        return gen(self.model, messages)

    def generate_environment(self, feature: Feature):
        """Generate initial tabulated format for feature activations.

        Args:
            feature (Feature): The feature to explain, unnormalized activations

        Returns:
            environment (str): Initial tabulated format for feature activations
        """

        formatted_tokens = [f"{t}   {a}" for t, a in zip(feature.tokens, feature.acts)]
        return "\n".join(formatted_tokens)

    def action(self, messages: str, regenerate=False):
        """Given some explaination, generate phrases which maximize activations

        Args: 
            messages (str): Message history with the most recent being an explaination

        Returns:
            phrases (List[str]): Phrases which maximize activations
        """
        
        if regenerate:
            messages = messages[:-1]

        messages.append({
            "role" : "user",
            "content" : ACTION_PROMPT
        })

        return gen(self.model, messages)

    def parse_action(self, phrase):
        """Given some phrases, parse them into scorable tokens for the RewardEngine

        Args:
            phrase (str): Phrase which maximize activations

        Returns:
            parsed_phrase (str): Parsed phrase
        """
        pattern = r"\{(.+?)\}"
        parsed_phrase = re.findall(pattern, phrase)
        if not parsed_phrase:
            return False
        return parsed_phrase

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
        parsed_phrase = self.parse_action(action)

        while (not parsed_phrase):
            action = self.action(explaination, regenerate=True)
            parsed_phrase = self.parse_action(action)

        reward = self.reward_engine.score(feature, parsed_phrase)
        self.scores.append(reward)

        reflection = self.reflect(reward)

        self.trial[0] = reflection

class RewardEngine:

    def __init__(self, model, dictionaries):
        self.model = model
        self.dictionaries = dictionaries

    def score(self, feature, parsed_phrase):
        activation = self.get_activation(feature.location)

        # TODO: SOME SCORING
        return 0

    def get_activation(self, location): 

        with torch.no_grad():
            acts = self.model.get_activation(location.layer)
        
        return acts[location.index]
