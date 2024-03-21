import re
from typing import List

from .utils import gen, Feature
from .prompts import SYSTEM_PROMPT, ACTION_PROMPT, RE_EXPLAIN_PROMPT

class Agent:

    def __init__(
            self, 
            model, 
            mem: List,
        ):
        self.model = model
        self.tokenizer = model.tokenizer

        self.mem = mem

    def explain(
            self, 
            features: List[Feature]
        ) -> None:
        """Generate detailed explanations for a list of neurons.

        Args:
            features (List[Feature]): List of features to explain.
        """

        question = SYSTEM_PROMPT.format(
            samples = self.build_activations(features)
        )

        if len(self.mem) == 1:

            environment = {
                "role" : "user",
                "content" : question
            }
         
            self.mem[-1].agent.append(environment)
        else: 
            

            re_explain = {
                "role" : "user",
                "content" : RE_EXPLAIN_PROMPT.format(
                    reflections = self.get_reflections(),
                    question = question
                )
            }

            self.mem[-1].agent.append(re_explain)

        explaination = {
            "role" : "assistant",
            "content" : gen(self.model, self.mem[-1].agent)  
        }
        
        self.mem[-1].agent.append(explaination)

    def get_reflections(self) -> str:
        """Format reflections from the previous trial

        Args:
            None

        Returns:
            str: Formatted reflections
        """

        reflections = ""
        for i, m in enumerate(self.mem[:-1]):
            refl = m.self_reflector[-1]["content"]
            reflections += f"Trial {i} Reflection:\n{refl}\n"
        
        return reflections
    
    def build_activations(
            self, 
            features: List[Feature]
        ) -> str:
        """Format the activations for a list of features.

        Args:
            features (List[Feature]): List of features to format.

        Returns:
            str: Formatted activations
        """

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

    def action(self) -> List[str]:
        """Generate a list of phrases which would maximize the activations of a neuron.
        
        Args:
            None

        Returns:
            List[str]: List of phrases
        """

        self.mem[-1].agent.append({
            "role" : "user",
            "content" : ACTION_PROMPT
        })

        generated_action = gen(self.model, self.mem[-1].agent)
        
        while not self.parse_action(generated_action):
            generated_action = gen(self.model, self.mem[-1].agent)

        self.mem[-1].agent.append({
            "role" : "assistant",
            "content" : generated_action
        })

        return self.parse_action(generated_action)

    def parse_action(
            self, 
            phrase: str
        ) -> List[str]:
        """Parse phrases from model.

        Args:
            phrase (str): Phrase to parse

        Returns:
            List[str]: Parsed phrases
        """

        pattern = r'\[(.*?)\]'
        matches = re.findall(pattern, phrase)

        if len(matches) != 3:
            return []
        
        return matches
        
    def __call__(
            self, 
            features: List[Feature]
        ) -> List[str]:
        """Run the agent.

        Args:
            features (List[Feature]): List of features to explain.

        Returns:
            List[str]: List of phrases
        """

        self.explain(features)
        return self.action()

