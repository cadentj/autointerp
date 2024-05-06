from dataclasses import dataclass
from utils import topk
from tqdm import tqdm
import torch as t
from utils import gen
from detection_prompts import get_detection_template

from dataclasses import field
@dataclass
class DetectionScorerConfig:
    max_tokens: int = 1000
    temperature: float = 0.0

    n_real: int = 10
    n_batches: int = 2
    real_ids : list = field(default_factory=list)

    left_ctx: int = 15
    right_ctx: int = 2

    verbose : bool = False

class DetectionScorer:

    def __init__(
        self,
        model,
        state,
        cfg: DetectionScorerConfig = None
    ):
        self.model = model
        self.state = state
    
        if cfg is None:
            cfg = DetectionScorerConfig()
        self.cfg = cfg

    def get_mixed_examples_list(self, feature_id):
        n_real = self.cfg.n_real
        batch_size = 2
        n_batches = self.cfg.n_batches
        real_ids = self.cfg.real_ids
        left_ctx = self.cfg.left_ctx
        right_ctx = self.cfg.right_ctx

        n_examples = n_real * n_batches

        assert (len(real_ids) == n_real)
        _, top_inds = topk(self.state.act_cache[:,:, feature_id], n_examples)

        true_examples_list = []

        for batch, pos in top_inds:
            start_pos = max(0, pos - left_ctx)
            end_pos = min(self.state.tok_cache.size(1), pos + right_ctx + 1)
            true_example_toks = self.state.tok_cache[batch, start_pos : end_pos]
            true_examples_list.append(true_example_toks)

        mixed_examples_list = []
        while len(mixed_examples_list) < n_examples:
            b = t.randint(0, self.state.act_cache.size(0), [1]).squeeze()
            s = t.randint(left_ctx, self.state.tok_cache.size(1) - right_ctx, [1]).squeeze()

            fake_example_toks = self.state.tok_cache[b, s : s + right_ctx + 1]
            fake_example_acts = self.state.act_cache[b, s : s + right_ctx + 1, feature_id]

            if sum(fake_example_acts) < 0.01:
                mixed_examples_list.append(fake_example_toks)

        for b in range(n_batches):
            for i in range(n_real):
                mixed_examples_list[b*batch_size + real_ids[i]] = true_examples_list[b*n_real + i]


        mixed_examples_list = self.model.tokenizer.batch_decode(mixed_examples_list)
        mixed_examples_list = [example.replace("{", "{{").replace("}", "}}") for example in mixed_examples_list]

        return mixed_examples_list