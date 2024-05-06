from dataclasses import dataclass
from tqdm import tqdm
import torch as t
from utils import gen
from prompting import get_detection_template

import re


from dataclasses import field

@dataclass
class DetectionScorerConfig:
    max_tokens: int = 1000
    temperature: float = 0.0

    n_real: int = 10
    n_batches: int = 2
    real_ids : list = field(default_factory=lambda: [0, 2, 5, 6, 9, 10, 11, 12, 18, 19])

    l_ctx: int = 15
    r_ctx: int = 2

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
        batch_size = 2

        n_examples = self.cfg.n_real * self.cfg.n_batches

        assert (len(self.cfg.real_ids) == self.cfg.n_real)

        true_examples_list = self.state.examples[n_examples]

        mixed_examples_list = []

        while len(mixed_examples_list) < n_examples:
            b = t.randint(0, self.state.act_cache.size(0), [1]).squeeze()
            s = t.randint(self.cfg.l_ctx, self.state.tok_cache.size(1) - self.cfg.r_ctx, [1]).squeeze()

            fake_example_toks = self.state.tok_cache[b, s : s + self.cfg.r_ctx + 1]
            fake_example_acts = self.state.act_cache[b, s : s + self.cfg.r_ctx + 1, feature_id]

            if sum(fake_example_acts) < 0.01:
                mixed_examples_list.append(fake_example_toks)

        for b in range(self.cfg.n_batches):
            for i in range(self.cfg.n_real):
                mixed_examples_list[b*batch_size + self.cfg.real_ids[i]] = true_examples_list[b*self.cfg.n_real + i]


        mixed_examples_list = self.model.tokenizer.batch_decode(mixed_examples_list)
        mixed_examples_list = [example.replace("{", "{{").replace("}", "}}") for example in mixed_examples_list]

        return mixed_examples_list

    def score(self, feature_id, explanation_list):
        self.get_mixed_examples_list(feature_id)
        score_list = []

        for explanation in tqdm(explanation_list):
            summed_detection_rate = 0.0
            summed_false_pos_rate = 0.0

        for b in range(self.cfg.n_batches):

            mixed_examples_list = self.mixed_examples_list[b*2*self.cfg.n_real : (b+1)*2*self.cfg.n_real]
            mixed_examples_str = ''
            for i in range(len(mixed_examples_list)):
                mixed_examples_str += f'Example {i+1}: {mixed_examples_list[i]}\n'


            detection_rate, false_pos_rate = self.query(explanation, mixed_examples_str)

            summed_detection_rate += detection_rate
            summed_false_pos_rate += false_pos_rate

        detection_rate = summed_detection_rate /  self.cfg.n_batches
        false_pos_rate = summed_false_pos_rate /  self.cfg.n_batches
        score_list.append((detection_rate, false_pos_rate))

        return score_list

    def query(self, explanation, mixed_examples_str):
        prompt = {
            "prompt": mixed_examples_str,
            "prompt_template": get_detection_template(mixed_examples_str, explanation),
            "max_tokens" : 1000,
            "temperature" : 0.0
        }

        output = gen(prompt)

        output_str = ''
        for i in output:
            output_str += i

        nums = self.extract_numbers_from_string(output_str)

        detection_rate = sum([num-1 in self.cfg.real_ids for num in nums]) / self.cfg.n_real
        false_pos_rate = sum([num-1 not in self.cfg.real_ids for num in nums]) / self.cfg.n_real

        return detection_rate, false_pos_rate


    def extract_numbers_from_string(s):
        # Use regex to find all occurrences of numbers between square brackets
        match = re.search(r'\[(.*?)\]', s)
        if not match:
            return []
        
        # Extract the substring that contains the numbers
        numbers_str = match.group(1)
        # Use a regex to find all numbers in the substring
        numbers = [int(num) for num in re.findall(r'-?\d+', numbers_str)]
        
        return numbers


    