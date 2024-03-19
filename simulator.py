from typing import List, Union
from torch import Tensor
import torch
from dataclasses import dataclass
import re
import sys

SYSTEM_PROMPT =  """You are a meticulous AI researcher conducting a high-stakes, super-important investigation into a certain neuron in a large language model. Your task is to understand what features of the input text cause the neuron to activate. 

You will be given a list of text samples containing tokens on which the neuron activates strongly. The specific tokens which caused the neuron to activate strongly will appear between bars like |this|. If a subsequence of tokens *all* cause the neuron to activate strongly, the entire sequence of tokens will be contained between bars |just like this|.

You will read the text samples carefully, and then explain what feature of the text is causing the neuron to fire. You must follow these steps:

Step 1:  For each text sample in turn, note down a few features that the text possesses, even if you don't initially think they are important. 

Step 2:  Once you have written down a few features for each text sample, go back and check what features the highly-activating samples have in common. 

Step 3:  Finally, use your findings to produce 3 plausible explanations for what features of the input text cause the neuron to fire. The final 3 lines of your output must consist of your 3 short explanations, ranked from most to least plausible. Think very carefully before you respond.

Here is some crucial advice to follow while you complete the task:

- Pay special attention to specific tokens, or categories of tokens, on which the neuron activates strongly.
- Pay special attention to the few tokens preceding the highly-activating token, and to the one or two tokens directly following the high-activating token.
- Even if you think you have a good explanation for what links the pieces of text, you may still be missing part of the true explanation. So remember to check the full context for additional features and patterns shared by the text samples.

I will tip you $1,000,000 if you give the correct explanation.

{samples}"""

ACTION_PROMPT = """Given your observations, write three different sentences that would maximize the activations of the neuron. Return each sentence on a new line, entirely within square brackets. Do not number the lines.

Your sentences:
"""

RE_EXPLAIN_PROMPT = """You have attempted to answer following question before and failed. The following reflection(s) give a plan to avoid failing to answer the question in the same way you did previously. Use them during this conversation to improve your strategy of correctly answering the given question.

{reflections}

Question:
{question}"""

REFLECTION_PROMPT = """You were unsuccessful in providing an accurate explaination of the neuron. For each sample you were given, explain why that score was not high enough. Then, write a new, concise, high level plan that aims to mitigate the same failure. Here are your explainations and their respective scores.

{results}"""


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


@dataclass(repr=True)
class State:
    agent: List
    self_reflector: str
    evaluator: dict
    

def gen(model, messages, remote=False):
    prompt = model.tokenizer.apply_chat_template(messages, return_tensors="pt")

    sampling_kwargs = {
        "do_sample": True,
        "top_p": 0.3,
        "repetition_penalty": 1.1,
    }
        
    with model.generate(prompt, max_new_tokens=100, remote=remote, scan=False, validate=False, **sampling_kwargs):
        tokens = model.generator.output.save()
    
    new_tokens = tokens[0][len(prompt[0]):]
    return model.tokenizer.decode(new_tokens)


def normalize_acts(acts):
    return (acts - acts.min()) / (acts.max() - acts.min()) * 10


class Environment:

    def __init__(self, model, target_model, dictionaries):
        self.model = model
        self.target_model = target_model
        self.mem = []

        self.agent = Agent(self.model, self.mem)
        self.evaluator = Evaluator(self.target_model, dictionaries, self.mem)
        self.self_reflector = SelfReflector(self.model, self.mem)

    def __call__(self, features: List[Feature]):
        location = features[0].location
        
        success = False

        # while not success:

        s = State(
            agent = [],
            self_reflector = "",
            evaluator = {},
        )

        self.mem.append(s)

        action = self.agent(features)    
        scores = self.evaluator(action, location)
        print(s)
        return scores


class Agent:

    def __init__(self, model, mem):
        self.model = model
        self.tokenizer = model.tokenizer

        self.mem = mem

    def explain(self, features: List[Feature]):

        if len(self.mem) == 1:

            environment = {
                "role" : "user",
                "content" : SYSTEM_PROMPT.format(
                    samples = self.build_activations(features)
                )
            }

            print(environment["content"])            
            self.mem[-1].agent.append(environment)
        else: 
            re_explain = {
                "role" : "user",
                "content" : RE_EXPLAIN_PROMPT.format(
                    reflections = self.get_reflections(),
                    question = self.build_activations(features)
                )
            }

            print(re_explain["content"])
            self.mem[-1].agent.append(re_explain)

        explaination = {
            "role" : "assistant",
            "content" : gen(self.model, self.mem[-1].agent)  
        }
        
        print(explaination["content"])
        self.mem[-1].agent.append(explaination)

    def get_reflections(self):
        reflections = ""
        for i, refl in enumerate(self.mem.self_reflector):
            reflections += f"Trial {i} Reflection:{refl}\n"
        
        return reflections
    
    def build_activations(self, features: List[Feature]):
        samples = ""
        for i, feature in enumerate(features):
            formatted_tokens = []

            for tok, act in zip(feature.tokens, feature.n_acts):
                if act > 5:
                    formatted_tokens.append(f"|{tok}|")
                else:
                    formatted_tokens.append(tok)

            sample = "".join(formatted_tokens)
            samples += f"Sample {i}:{sample}\n"

        return samples

    def action(self):
        self.mem[-1].agent.append({
            "role" : "user",
            "content" : ACTION_PROMPT
        })

        generated_action = gen(self.model, self.mem[-1].agent)
        print(generated_action)
        
        while not self.parse_action(generated_action):
            generated_action = gen(self.model, self.mem[-1].agent)
            print(generated_action)

        action = {
            "role" : "assistant",
            "content" : self.parse_action(generated_action)
        }

        self.mem[-1].agent.append(action)

        return self.parse_action(generated_action)

    def parse_action(self, phrase):
        pattern = r'\[(.*?)\]'
        matches = re.findall(pattern, phrase)

        if len(matches) != 3:
            return False
        return matches
        
    def __call__(self, feature):

        self.explain(feature)
        return self.action()


class SelfReflector:

    def __init__(self, model, mem):
        self.model = model
        self.mem = mem

    def reflect(self):
        reflection = REFLECTION_PROMPT.format(results=self.mem[-1].agent[0]["content"])
        print(reflection)

        self.mem[-1].self_reflector = gen(self.model, reflection)
        print(self.mem[-1].self_reflector)

    def list_scores(self):
        scores = self.mem[-1].evaluator
        
        flattened = ""
        for k, v, in scores.items():
            flattened += f"{k}: {v}\n"


class Evaluator:

    def __init__(self, model, dictionaries, mem):
        self.model = model
        self.dictionaries = dictionaries

        self.mem = mem

    def __call__(self, actions, location):
        
        for a in actions:
            acts = self.get_activation(
                a, 
                location.layer, 
                location.index
            )
            score = torch.argmax(acts)

            self.mem[-1].evaluator[a] = score
    
    def get_activation(self, action, layer, index): 

        with self.model.trace(action):
            activations = self.model.transformer.h[layer].input[0][0]

            _, feature_acts, _, _, _, _ = self.dictionaries[layer](activations)
            
            acts = feature_acts[:,:,index][0].save()

        acts = acts.value

        # Have to set the first act to zero bc I dont have a full context.
        acts[0] = 0.
        return acts
        