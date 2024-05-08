from dataclasses import dataclass
import threading
from threading import Lock
import os

import torch as t
from torch import Tensor
from tqdm import tqdm
from datasets import load_dataset, Dataset
from transformer_lens import utils

from utils import topk, log, log_conversation
from prompting import get_simple_condenser_template
from config import EnvConfig, ExplainerConfig, CondenserConfig, DetectionScorerConfig, GenerationScorerConfig
from explainer import Explainer
from condenser import Condenser
from detection_scorer import DetectionScorer
from generation_scorer import GenerationScorer

@dataclass
class ThreadState:
    history: list
    examples: list
    random_examples: list
    max_act: float
    layer: int
    feature_id: int
    lock: Lock = None

    def __post_init__(self):
        self.lock = Lock()  # Initialize the lock when a ThreadState object is created


class Environment: 
    def __init__(
        self,
        model, 
        sae,
        cfg: EnvConfig,
        provider: str,
    ):  
        self.model = model
        self.sae = sae
        self.cfg = cfg
        self.provider = provider
        self.lock = Lock()


    def execute(
        self,
        feature_id: int, 
        explainer_cfg: ExplainerConfig, 
        condenser_cfg: CondenserConfig, 
        d_scorer_cfg: DetectionScorerConfig, 
        gen_scorer_cfg: GenerationScorerConfig
    ):
        with self.lock:
            state = self.load_id(feature_id, explainer_cfg, condenser_cfg, d_scorer_cfg, gen_scorer_cfg)

        self.run_feature(state)
        

    def load_cache(
        self,
        layer: int, # sae layer
    ):
        self.seed = self.cfg.seed
        self.layer = layer

        tokenized_data = self.load_webtext(self.cfg.batch_len)

        self.tok_cache, self.act_cache = self.load_features(
            tokenized_data,
            layer,
            num_batches=self.cfg.num_batches,
            minibatch_size=self.cfg.minibatch_size
        )

    def load_id(
        self,
        feature_id: int, # index of feature
        explainer_cfg: ExplainerConfig,
        condenser_cfg: CondenserConfig,
        d_scorer_cfg: DetectionScorerConfig,
        gen_scorer_cfg: GenerationScorerConfig,
    ):

        examples, max_act = self.get_top_examples(feature_id)
        random_examples = self.generate_random_examples(feature_id, d_scorer_cfg)

        state = ThreadState(
            history=[],
            examples=examples,
            random_examples=random_examples,
            max_act=max_act,
            layer = self.layer,
            feature_id=feature_id
        )

        self.explainer = Explainer(self.model, state, cfg=explainer_cfg)
        self.condenser = Condenser(cfg=condenser_cfg)
        self.d_scorer = DetectionScorer(self.model, state, cfg=d_scorer_cfg)
        self.gen_scorer = GenerationScorer(self.model, state, cfg=gen_scorer_cfg)

        return state
        

    def load_webtext(self, batch_len) -> Dataset:
        """Load OpenWebText. Uses `tokenize_and_concatenate` to split
        the large corpus, tokenize, reshape for parallelization, and concatenate.

        Args:
            batch_len (int): The number of tokens to load.

        Returns:
            Dataset: The tokenized dataset.
        """

        data = load_dataset("stas/openwebtext-10k", split="train")
        tokenized_data = utils.tokenize_and_concatenate(data, self.model.tokenizer, max_length=128)
        tokenized_data = tokenized_data.shuffle(batch_len)
        return tokenized_data


    def get_top_examples(
        self,
        feature_id, 
    ):
        examples_list = []

        top_acts, top_inds = topk(self.act_cache[:,:, feature_id], self.cfg.n_examples)
        max_act = top_acts[0]

        batch_len = self.cfg.batch_len

        for batch, pos in top_inds:
            start_pos = max(0, pos - self.cfg.l_ctx)
            end_pos = min(batch_len, pos + self.cfg.r_ctx + 1)
            example_toks = self.tok_cache[batch, start_pos : end_pos]
            example_acts = self.act_cache[batch, start_pos : end_pos, feature_id]

            example = (example_toks, example_acts)
            
            examples_list.append(example)

        return examples_list, max_act


    def generate_random_examples(self, feature_id, d_scorer_cfg):
        mixed_examples_list = []

        while len(mixed_examples_list) < d_scorer_cfg.n_batches*d_scorer_cfg.batch_size:
            # Choose a random batch and sequence position
            batch = t.randint(0, self.act_cache.size(0), [1]).squeeze()
            pos = t.randint(self.cfg.l_ctx, self.tok_cache.size(1) - self.cfg.r_ctx, [1]).squeeze()

            # Extract the tokens and activations for the fake example
            fake_example_toks = self.tok_cache[batch, pos - self.cfg.l_ctx : pos + self.cfg.r_ctx + 1]
            fake_example_acts = self.act_cache[batch, pos - self.cfg.l_ctx : pos + self.cfg.r_ctx + 1, feature_id]

            # Check the fake example does not activate the real one
            if sum(fake_example_acts) < 0.01:
                # Append the fake example to the list
                mixed_examples_list.append(fake_example_toks)

        return mixed_examples_list


    def load_features(
        self,
        tokenized_data,
        layer,
        num_batches=1000, 
        minibatch_size=20
    ):
        # get however many tokens we need
        toks = tokenized_data["tokens"][:num_batches]

        n_mini_batches = len(toks) // minibatch_size

        tok_batches = [
            toks[minibatch_size*i : minibatch_size*(i+1), :] 
            for i in range(n_mini_batches)
        ]

        feature_acts = None 

        for batch in tqdm(tok_batches):

            with self.model.trace(batch):
                activations = self.model.transformer.h[layer].input[0][0]

                middle = self.sae(activations)

                acts = middle[1]
                acts.save()

            acts = acts.value.detach().cpu()

            if feature_acts is None:
                feature_acts = acts
            
            else:
                feature_acts = t.cat([feature_acts, acts], dim=0)

            del acts

        feature_acts[:, 0, :] = 0

        print("Activation Cache Size:", feature_acts.size())

        return toks, feature_acts


    def run_feature(self, state: ThreadState):
        long_explanation_list = self.explainer.explain()

        log(state, "system", long_explanation_list)
        print("Explainer completed.")

        explanation_list, output = self.condenser.condense(long_explanation_list, return_output=True)

        log(state, "section", "Running condenser.")
        log(state, "user", get_simple_condenser_template(long_explanation_list))
        log(state, "assistant", output)
        print("Condenser completed.")

        d_scores_list = self.d_scorer.score(explanation_list)
        print("Detection Scorer completed.")

        g_scores_list = self.gen_scorer.score(explanation_list, self.sae)
        print("Generation Scorer completed.")

        results = ""

        for i in range(len(explanation_list)):
            results += f"Explanation {i}:\n"
            results += f"Detection Score: {d_scores_list[i]}\n"
            results += f"Generation Score: {g_scores_list[i]}\n"
            results += f"Explanation: {explanation_list[i]}\n\n"

        log(state, "section", "Results.")
        log(state, "system", results)

        save_path = f"./results/{self.provider}/{state.layer}_{state.feature_id}.txt"

        log_conversation(state.history, save_path)



