import re

from tqdm import tqdm
import torch as t

from utils import gen, log
from prompting import get_detection_template, get_simple_detection_template
from config import DetectionScorerConfig


class DetectionScorer:

    def __init__(
        self,
        model,
        state,
        cfg: DetectionScorerConfig
    ):
        self.model = model
        self.state = state
        self.cfg = cfg


    def get_mixed_examples_list(self):

        n_examples = self.cfg.n_real * self.cfg.n_batches

        assert (len(self.cfg.real_ids) == self.cfg.n_real)

        true_examples_list = [i[0] for i in self.state.examples[:n_examples]]

        mixed_examples_list = self.state.random_examples

        for batch in range(self.cfg.n_batches):
            for i in range(self.cfg.n_real):
                mixed_examples_list[batch*self.cfg.batch_size + self.cfg.real_ids[i]] = true_examples_list[batch*self.cfg.n_real + i]

        mixed_examples_list = self.model.tokenizer.batch_decode(mixed_examples_list)
        mixed_examples_list = [example.replace("{", "{{").replace("}", "}}") for example in mixed_examples_list]

        return mixed_examples_list


    def score(self, explanation_list):
        mixed_examples_list = self.get_mixed_examples_list()
        score_list = []

        log(self.state, "section", "Running detection scoring.")
        log(self.state, "user", get_simple_detection_template("<EXPLANATION>", "<MIXED EXAMPLES>"))

        for explanation in tqdm(explanation_list):
            summed_detection_rate = 0.0
            summed_false_pos_rate = 0.0

            log(self.state, "system", f"Running on explanation: {explanation}")

            for b in range(self.cfg.n_batches):
                log(self.state, "system", f"Processing batch {b+1} of {self.cfg.n_batches}")

                mixed_examples = mixed_examples_list[b*2*self.cfg.n_real : (b+1)*2*self.cfg.n_real]
                mixed_examples_str = ''
                for i in range(len(mixed_examples)):
                    mixed_examples_str += f'Example {i+1}: {mixed_examples[i]}\n'

                detection_rate, false_pos_rate = self.query(explanation, mixed_examples_str)

                summed_detection_rate += detection_rate
                summed_false_pos_rate += false_pos_rate

            detection_rate = summed_detection_rate /  self.cfg.n_batches
            false_pos_rate = summed_false_pos_rate /  self.cfg.n_batches
            score_list.append((detection_rate, false_pos_rate))

        return score_list


    def query(self, explanation, mixed_examples_str):
  
        generation_kwargs = {
            "max_tokens":self.cfg.max_tokens,
            "temperature":self.cfg.temperature
        }
        
        nums, output = gen(
            get_detection_template(mixed_examples_str, explanation), 
            postprocess = self.extract_numbers_from_string,
            generation_kwargs=generation_kwargs
        )
 
        log(self.state, "assistant", output)

        detection_rate = sum([num-1 in self.cfg.real_ids for num in nums]) / self.cfg.n_real
        false_pos_rate = sum([num-1 not in self.cfg.real_ids for num in nums]) / self.cfg.n_real

        return detection_rate, false_pos_rate


    def extract_numbers_from_string(self, s):
        # Use regex to find all occurrences of numbers between square brackets
        match = re.search(r'\[(.*?)\]', s)
        if not match:
            return []
        
        # Extract the substring that contains the numbers
        numbers_str = match.group(1)
        # Use a regex to find all numbers in the substring
        numbers = [int(num) for num in re.findall(r'-?\d+', numbers_str)]
        
        return numbers


    