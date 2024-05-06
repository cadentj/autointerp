from dataclasses import dataclass

import torch as t
from torch import Tensor
from tqdm import tqdm
from datasets import load_dataset, Dataset
from transformer_lens import utils

from utils import topk
from explainer import Explainer, ExplainerConfig
from detection_scorer import DetectionScorer, DetectionScorerConfig
from generation_scorer import GenerationScorer, GenerationScorerConfig


@dataclass
class EnvConfig:
    num_batches: int = 2_000
    minibatch_size: int = 150
    seed: int = 22
    batch_len: int = 128
    n_examples = 40
    l_ctx: int = 15
    r_ctx: int = 4


@dataclass
class State:
    act_cache: Tensor
    tok_cache: Tensor
    history: list
    examples: list
    max_act: float
    layer: int
    feature_id: int


class Environment: 
    def __init__(
        self,
        model, 
        sae_list,
    ):  
        self.model = model
        self.sae_list = sae_list
    

    def load(
        self,
        layer: int, # sae layer
        feature_id: int, # index of feature
        cfg: EnvConfig = None,
        explainer_cfg: ExplainerConfig = None,
        detection_cfg: DetectionScorerConfig = None,
        generation_cfg: GenerationScorerConfig = None,
    ):
        if cfg is None:
            cfg = EnvConfig()

        self.cfg = cfg
        self.seed = cfg.seed

        tokenized_data = self.load_webtext(cfg.batch_len)

        tok_cache, act_cache = self.load_features(
            tokenized_data,
            layer,
            num_batches=cfg.num_batches,
            minibatch_size=cfg.minibatch_size
        )

        examples, max_act = self.get_top_examples(act_cache, tok_cache, feature_id)

        self.state = State(
            act_cache=act_cache,
            tok_cache=tok_cache,
            history=[],
            examples=examples,
            max_act=max_act,
            layer = layer,
            feature_id=feature_id
        )

        self.explainer = Explainer(self.model, self.state, cfg=explainer_cfg)
        self.d_scorer = DetectionScorer(self.model, self.state, cfg=detection_cfg)
        self.gen_scorer = GenerationScorer(self.model, self.state, cfg=generation_cfg)
        

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
        act_cache,
        tok_cache,
        feature_id, 
    ):
        examples_list = []

        top_acts, top_inds = topk(act_cache[:,:, feature_id], self.cfg.n_examples)
        max_act = top_acts[0]

        batch_len = self.cfg.batch_len

        for batch, pos in top_inds:
            start_pos = max(0, pos - self.cfg.l_ctx)
            end_pos = min(batch_len, pos + self.cfg.r_ctx + 1)
            example_toks = tok_cache[batch, start_pos : end_pos]
            example_acts = act_cache[batch, start_pos : end_pos, feature_id]

            example = (example_toks, example_acts)
            
            examples_list.append(example)

        return examples_list, max_act


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

                middle = self.sae_list[layer](activations)

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


    def run(self):
        self.explainer.generate_top_examples(7000)
        
