from dataclasses import dataclass
from utils import topk

@dataclass
class ExplainerConfig:

    max_tokens : int = 2000
    temperature : float = 0.5

    batch_size: int = 10
    n_batches : int = 2
    runs_per_batch: int = 1

    left_ctx: int = 15
    right_ctx: int = 2
    activation_threshold: float = 0.4,

    verbose: bool = False

# delimiters
l = '<<'
r = '>>'


class Explainer:

    def __init__(self, cfg):

        for key, value in cfg.as_dict().items():
            setattr(self, key, value)

        self.n_examples = self.batch_size * self.n_batches

    def fix(self, string):
        string = string.replace(r+l, "")
        string = string.replace("{", "{{").replace("}", "}}")
        return string

    def generate_top_examples(self, FEATURE_ACTS, TOKS, feature_id, tokenizer):

        top_acts, top_inds = topk(FEATURE_ACTS[:,:, feature_id], self.n_examples)
        max_act = top_acts[0]

        top_examples_list = []

        batch_size = TOKS.size(1)

        for i, (batch, tok) in enumerate(top_inds):

            start = max(0, tok - self.left_ctx)
            end = min(batch_size, tok + self.right_ctx)

            example_toks = TOKS[batch, start:end]

            example_acts = FEATURE_ACTS[batch, start:end, feature_id]

            delimited_string = ''
            for pos in range(example_toks.size(0)):
                if example_acts[pos] > (self.activation_threshold * max_act):
                    delimited_string += l + tokenizer.decode(example_toks[pos]) + r
                else:
                    delimited_string += tokenizer.decode(example_toks[pos])

            delimited_string = self.fix(delimited_string)

            top_examples_list.append(delimited_string)

        return top_examples_list

    