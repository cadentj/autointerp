from typing import List
from torch import Tensor
import torch
from dataclasses import dataclass
import re
from prompts import SYSTEM_PROMPT, ACTION_PROMPT, RE_EXPLAIN_PROMPT, REFLECTION_PROMPT

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
    prompt = model.tokenizer.apply_chat_template(messages, return_tensors="pt")

    with model.generate(prompt, max_new_tokens=100, remote=remote, scan=False, validate=False):
        tokens = model.generator.output.save()

    return model.tokenizer.decode(tokens[0])

class Agent:

    def __init__(self, model, reward_engine):
        self.model = model
        self.tokenizer = model.tokenizer

        self.reward_engine = reward_engine

        self.scores = []
        self.trials = []

    def explain(self, feature: Feature):
        """Generate a detailed explaination of feature activations

        Args:
            feature (Feature): The feature to explain, unnormalized activations

        Returns:
            explaination (str): A detailed explaination of the feature activations
        """
        
        if not self.scores:
            environment = {
                "role" : "user",
                "content" : self.generate_environment(feature)
            }

            messages = [environment]
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
        activations = "\n".join(formatted_tokens)

        return SYSTEM_PROMPT.format(sentence="abacdqei", activations=activations)

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

        flattened = flatten_conversation(self.trails[-1])

        return REFLECTION_PROMPT.format(trial=flattened)
        
        
    def __call__(self, feature: Feature):

        # LOOP THIS
        explaination = self.explain(feature)

        return explaination

        # action = self.action(explaination)
        # parsed_phrase = self.parse_action(action)

        # while (not parsed_phrase):
        #     action = self.action(explaination, regenerate=True)
        #     parsed_phrase = self.parse_action(action)

        # reward = self.reward_engine.score(feature, parsed_phrase)
        # self.scores.append(reward)

        # reflection = self.reflect(reward)

        # self.trials.append(reflection)

def flatten_conversation(conversation):
    flattended = ""

    for message in conversation:
        if message["role"] == "user":
            flattended += f"(USER)\n{message['content']}\n\n"
        else:
            flattended += f"(MODEL)\n{message['content']}\n\n"

    return flattended[:-2]

class RewardEngine:

    def __init__(self, model, dictionaries, threshold=8):
        self.model = model
        self.dictionaries = dictionaries

    def score(self, feature, parsed_phrase):
        activation = self.get_activation(feature.location)

        # TODO: SOME SCORING
        return 0

    def get_activation(self, location, phra): 

        # with torch.no_grad():

        #     with self.model.invoke()
        
        # return acts[location.index]

        pass

