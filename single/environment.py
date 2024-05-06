from dataclasses import dataclass
from torch import Tensor
from tqdm import tqdm
import torch as t
from datasets import load_dataset, Dataset
from transformer_lens import utils
from detection_scorer import DetectionScorer

from explainer import Explainer

@dataclass
class EnvConfig:
    num_batches: int = 2_000
    minibatch_size: int = 150
    seed: int = 22
    batch_len: int = 128

@dataclass
class State:
    act_cache: Tensor
    tok_cache: Tensor
    history: list

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
        layer, # sae layer
        feature_id: int, # index of feature
        cfg: EnvConfig = None
    ):
        if cfg is None:
            cfg = EnvConfig()

        self.seed = cfg.seed

        tokenized_data = self.load_webtext(cfg.batch_len)

        tok_cache, act_cache = self.load_features(
            tokenized_data,
            layer,
            num_batches=cfg.num_batches,
            minibatch_size=cfg.minibatch_size
        )

        self.state = State(
            act_cache=act_cache,
            tok_cache=tok_cache,
            history=[]
        )

        self.explainer = Explainer(self.model, self.state)
        self.d_scorer = DetectionScorer(self.model, self.state)
        

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
        
