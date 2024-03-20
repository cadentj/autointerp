from typing import List

from utils import log_conversation, State, Feature
from agent import Agent


class Environment:

    def __init__(self, model, target_model, dictionaries):
        self.model = model
        self.target_model = target_model
        self.mem = []

        self.agent = Agent(self.model, self.mem)
        self.evaluator = Evaluator(self.target_model, dictionaries, self.mem)
        self.self_reflector = SelfReflector(self.model, self.mem)

        self.past_key_values = None

    def render_state(self):

        for i, m in enumerate(self.mem):
            
            path = f"./logs/trial_{i}.log"
            conversation = m.agent
            conversation.append(m.self_reflector[-1])

            log_conversation(conversation, path)

    def __call__(self, features: List[Feature], max_trials=3):
        location = features[0].location

        for trial in range(max_trials):
            print(f"Trial {trial}")

            s = State(
                agent = [],
                self_reflector = [],
                evaluator = {},
            )

            self.mem.append(s)

            action = self.agent(features)  

            self.evaluator(action, location)

            if self.self_reflector():
                break

            self.render_state()

        return self.mem