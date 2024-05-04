from typing import List

from .Agent import Agent
from .Evaluator import Evaluator
from .SelfReflector import SelfReflector
from .Explainer import Explainer
from .utils import log_conversation, State, Feature, normalize_acts, Location

class Environment:

    def __init__(self, model, target_model, dictionaries, do_trials=False):
        self.model = model
        self.target_model = target_model
        self.mem = []
        
        self.dictionaries = dictionaries
        self.do_trials = do_trials

        self.agent = Agent(self.model, self.mem)
        self.evaluator = Evaluator(self.target_model, dictionaries, self.mem)
        self.self_reflector = SelfReflector(self.model, self.mem)
        self.explainer = Explainer(self.model, self.mem)

    def render_state(self) -> None:
        """Render the current state to a .log file.

        Args:
            None
        
        Returns:
            None
        """

        for i, m in enumerate(self.mem):
            
            path = f"./logs/trial_{i}.log"
            conversation = m.agent

            log_conversation(conversation, path)
            
    def __call__(
        self,
        prompts: List[str],
        layer: int,
        index: int,
        feature_type: str = "resid" # Other features not yet implemented
    ):
        location = Location(
            feature_type = feature_type,
            layer = layer,
            index = index
        )

        tokenizer = self.target_model.tokenizer

        features = []

        for prompt in prompts:
            
            # Without the full context, the first token often has high activations
            # I append a bos_token and remove it later to discard the high outlier
            prompt = tokenizer.bos_token + prompt
            tokens = tokenizer.encode(prompt)
            str_tokens = [tokenizer.decode(t) for t in tokens]

            with self.target_model.trace(tokens):
                activations = self.target_model.transformer.h[layer].input[0][0].save()

                middle = self.dictionaries[layer](activations)

                # See mats_sae_training/sae_training/spase_autoencoder.py
                # sae_out, feature_acts, loss, mse_loss, l1_loss, mse_loss_ghost_resid
                acts = middle[1][:,:,index][0].save()

            # Bos token activation discarded
            acts[0] = 0.
            acts = acts.value

            f = Feature(
                prompt = prompt,
                tokens = str_tokens,
                acts = acts,
                n_acts = normalize_acts(acts),
                location = location,
            )

            features.append(f)

        print("Loaded features")

        return self.run(features)


    def run(
            self, 
            features: List[Feature], 
        ) -> List[State]:
        """Run autointerp on a neuron, given a set of example Features.

        Args:
            features (List[Feature]): List of Features
            max_trials (int): Maximum number of trials to run
        
        Returns:
            List[State]: List of States
        """

        if not self.do_trials:
            return self.single_pass(features)
        else:
            return self.trials(features)

    def trials(
        self,
        features,
        max_trials=3
    ):
        location = features[0].location

        for trial in range(max_trials):
            print(f"Trial {trial}")

            s = State(
                agent = [],
                self_reflector = "",
                evaluator = {},
                kv = None,
            )

            self.mem.append(s)

            action = self.agent(features)  

            print(action)

            self.evaluator(action, location)

            if self.self_reflector():
                break

            self.render_state()

        return self.mem
    
    def single_pass(
        self,
        features: List[Feature]
    ):
        location = features[0].location

        s = State(
            agent = [],
            self_reflector = [],
            evaluator = {},
            kv = None,
        )

        self.mem.append(s)

        action = self.agent(features)  

        print(action)

        self.evaluator(action, location)

        self.self_reflector()

        result = self.explainer()
        
        self.render_state()

        del s.kv

        return result
