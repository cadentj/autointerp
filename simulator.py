from typing import List, Union
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

ACTION_PROMPT = """Given your observations, write three different sentences that would maximize the activations of the neuron. Return each sentence in brackets.

Example sentences:
[sentence1]
[sentence2]
[sentence3]

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

@dataclass
class State:
    agent: List
    self_reflector: str
    evaluator: Union[int, bool]

def gen(model, messages, remote=False):
    prompt = model.tokenizer.apply_chat_template(messages, return_tensors="pt")

    with model.generate(prompt, max_new_tokens=100, remote=remote, scan=False, validate=False):
        tokens = model.generator.output.save()
    
    new_tokens = tokens[0][len(prompt[0]):]
    return model.tokenizer.decode(new_tokens)
    
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

        self.agent = Agent(self.model, self.mem)
        self.self_reflector = SelfReflector(self.model, self.mem)
        self.evaluator = Evaluator(self.model, dictionaries, self.mem)

    def __call__(self, feature: Feature):
        
        success = False

        while not success:

            s = State(
                agent = [],
                self_reflector = "",
                evaluator = None,
            )

            self.mem.append(s)

            action = self.agent(feature)

            break

        return action

class Agent:

    def __init__(self, model, mem):
        self.model = model
        self.tokenizer = model.tokenizer

        self.mem = mem

    def explain(self, feature: Feature):

        if len(self.mem) == 1:

            environment = {
                "role" : "user",
                "content" : self.generate_environment(feature)
            }

            self.mem[-1].agent.append(environment)
        else: 
            re_explain = {
                "role" : "user",
                "content" : RE_EXPLAIN_PROMPT
            }

            self.mem[-1].agent.append(re_explain)


        explaination = {
            "role" : "assistant",
            "content" : gen(self.model, self.mem[-1].agent)  
        }
        
        self.mem[-1].agent.append(explaination)

        # print(explaination)
    
    def generate_environment(self, feature: Feature):

        formatted_tokens = [f"{t}   {a}" for t, a in zip(feature.tokens, feature.n_acts)]
        activations = "\n".join(formatted_tokens)

        return SYSTEM_PROMPT.format(sentence=feature.prompt, activations=activations)

    def action(self):
        self.mem[-1].agent.append({
            "role" : "user",
            "content" : ACTION_PROMPT
        })

        generated_action = gen(self.model, self.mem[-1].agent)
        
        while not self.parse_action(generated_action):
            generated_action = gen(self.model, self.mem[-1].agent)

        action = {
            "role" : "assistant",
            "content" : generated_action
        }

        self.mem[-1].agent.append(action)

        return generated_action

    def parse_action(self, phrase):
        pattern = r'\[(.*?)\]'
        matches = re.findall(pattern, phrase)

        print(phrase)
        print(matches)
        print(len(matches))
        
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

    def reflect(self, messages):
        return gen(self.model, messages)


class Evaluator:

    def __init__(self, model, dictionaries, mem):
        self.model = model
        self.dictionaries = dictionaries

        self.mem = mem

    def __call__(self, actions, location):
        scores = []
        for a in actions:
            acts = self.get_activation(
                a, 
                location.layer, 
                location.index
            )
            score = torch.argmax(acts)
            scores.append(score)

        return scores
    
    def get_activation(self, action, layer, index): 

        with self.model.trace(action):
            activations = self.model.transformer.h[layer].output[0]

            _, feature_acts, _, _, _, _ = self.dictionaries[layer](activations)
            
            acts = feature_acts[:,:,index].save()

        return acts
        