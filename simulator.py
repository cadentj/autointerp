from typing import List
from torch import Tensor
import torch
from dataclasses import dataclass
import re

SYSTEM_PROMPT =  """We're studying neurons in a neural network. Each neuron looks for some particular thing in a short document. Look at the parts of the document the neuron activates for and give a detailed explaination about what the neuron is doing. 

The activation format is token<tab>activation. Activation values range from 0 to 10. A neuron finding what it's looking for is represented by a non-zero activation value. The higher the activation value, the stronger the match.

Sentence: 
{sentence}

Activations:
<start>
{activations}
<end>

Here are three concise sentences that best describe the neurons behavior.
"""

ACTION_PROMPT = """Given your observations, write three sentences that would maximize the activations of the neuron. Return your sentences in brackets.

Example format:
{sentence1}
{sentence2}
{sentence3}

Your sentences:
"""

RE_EXPLAIN_PROMPT = "Given the self reflections..."

REFLECTION_PROMPT = """You will be given the history of a past experience in which you were placed in an environment and given a task to complete. You were unsuccessful in completing the task. Do not summarize your environment, but rather think about the strategy and path you took to attempt to complete the task. Devise a concise, new plan of action that accounts for your mistake with reference to specific actions that you should have taken. For example, if you tried A and B but forgot C, then devise a plan to achieve C with environment-specific actions. You will need this later when you are solving the same task. Give your plan after "Plan". Here are two examples:

{trial}

Given your score, """

@dataclass 
class Location:
    feature_type: str # mlp, resid, attn
    index: int # dictionary index
    layer: int # layer index

@dataclass
class Feature:
    prompt: str
    tokens: List[str]
    acts: Tensor
    n_acts: Tensor = None
    location: Location = None

def gen(model, messages, remote=False):
    prompt = model.tokenizer.apply_chat_template(messages, return_tensors="pt")

    with model.generate(prompt, max_new_tokens=100, remote=remote, scan=False, validate=False):
        tokens = model.generator.output.save()

    return model.tokenizer.decode(tokens[0])
    
def flatten_conversation(conversation):
    flattended = ""

    for message in conversation:
        if message["role"] == "user":
            flattended += f"(USER)\n{message['content']}\n\n"
        else:
            flattended += f"(MODEL)\n{message['content']}\n\n"

    return flattended[:-2]

class Environment:

    def __init__(self, model, dictionaries):
        self.model = model

        self.mem = []
        self.trials = []
    
    def load(self):

        self.self_reflector = SelfReflector(self.model)
        self.reward_engine = Evaluator(self.model, self.dictionaries)
        self.agent = Agent(self.model, self.reward_engine)

    def explain(self, feature: Feature):
        
        success = False

        while not success:
            action = self.agent(feature)

            


class Agent:

    def __init__(self, model, reward_engine):
        self.model = model
        self.tokenizer = model.tokenizer

        self.reward_engine = reward_engine

        self.scores = []
        self.trials = []

    def explain(self, feature: Feature):
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

        formatted_tokens = [f"{t}   {a}" for t, a in zip(feature.tokens, feature.n_acts)]
        activations = "\n".join(formatted_tokens)

        return SYSTEM_PROMPT.format(sentence=feature.prompt, activations=activations)

    def action(self, messages: str, regenerate=False):
        if regenerate:
            messages = messages[:-1]

        messages.append({
            "role" : "user",
            "content" : ACTION_PROMPT
        })

        return gen(self.model, messages)

    def parse_action(self, phrase):
        pattern = r"\{(.+?)\}"
        parsed_phrase = re.findall(pattern, phrase)
        if not parsed_phrase:
            return False
        return parsed_phrase
        
    def __call__(self, feature: Feature):

        explaination = self.explain(feature)

        action = self.action(explaination)
        parsed_phrase = self.parse_action(action)

        while (not parsed_phrase):
            action = self.action(explaination, regenerate=True)
            parsed_phrase = self.parse_action(action)

        return parsed_phrase

class SelfReflector:

    def __init__(self, model):
        self.model = model

    def reflect(self, messages):
        return gen(self.model, messages)


class Evaluator:

    def __init__(self, model, dictionaries, threshold=8):
        self.model = model
        self.dictionaries = dictionaries

    def evaluate(self, trials):

        flattened = flatten_conversation(trials[-1])

        prompt = REFLECTION_PROMPT.format(trial=flattened)

        return gen(self.model, prompt)

    def score(self, feature, parsed_phrase):
        acts = self.get_activation(feature.location)

        # TODO: SOME SCORING
        return 0

    def get_activation(self, location, phrase): 

        layer = location.layer
        index = location.index

        with self.model.trace(phrase):
            activations = self.model.transformer.h[layer].output[0]

            _, feature_acts, _, _, _, _ = self.dictionaries[layer](activations)
            
            acts = feature_acts[:,:,index].save()

        return acts
        