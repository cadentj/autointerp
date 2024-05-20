from dataclasses import dataclass
from typing import List

import torch as t
from tqdm import tqdm
from nnsight import LanguageModel

from datasets import load_dataset, Dataset
from transformer_lens import utils
from torch import Tensor

from .config import CacheConfig

@dataclass
class Example:
    text: str
    tokens: Tensor
    max_acts: Tensor

def unravel_index(flat_index, shape):

    indices = []
    for dim_size in reversed(shape):
        indices.append(flat_index % dim_size)
        flat_index = flat_index // dim_size
    return tuple(reversed(indices))


def topk(tensor, k):

    flat_tensor = tensor.flatten()

    top_values, flat_indices = t.topk(flat_tensor, k)

    original_indices = [unravel_index(idx.item(), tensor.size()) for idx in flat_indices]

    return top_values.tolist(), original_indices

class ActivationCache:

    def __init__(
        self,
        layer: int,
        model: LanguageModel,
        sae,
        cfg: CacheConfig
    ):  
    
        self.cfg = cfg
        self.model = model
        self.sae = sae
        self.load_cache(layer)

    def prepare_example(
        self,
        example_toks,
        example_acts,
        max_act
    ):
        delimited_string = ''
        for pos in range(example_toks.size(0)):
            if example_acts[pos] > (self.cfg.activation_threshold * max_act):
                delimited_string += self.cfg.l + self.model.tokenizer.decode(example_toks[pos]) + self.cfg.r
            else:
                delimited_string += self.model.tokenizer.decode(example_toks[pos])

        delimited_string = self.fix(delimited_string)

        return delimited_string
       
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
    
    def fix(self, string):
        string = string.replace(self.cfg.r+self.cfg.l, "")
        string = string.replace("{", "{{").replace("}", "}}")
        return string

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
    
    def load_cache(
        self,
        layer: int, # sae layer
    ):
        self.layer = layer

        tokenized_data = self.load_webtext(self.cfg.batch_len)

        self.tok_cache, self.act_cache = self.load_features(
            tokenized_data,
            layer,
            num_batches=self.cfg.num_batches,
            minibatch_size=self.cfg.minibatch_size
        )
    
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

            example = Example(
                text=self.prepare_example(
                    example_toks, 
                    example_acts, 
                    max_act
                ),
                tokens=example_toks,
                max_acts=example_acts
            )
            
            examples_list.append(example)

        return examples_list, max_act