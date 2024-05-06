import re

from tqdm import tqdm
import torch as t

from utils import gen, log
from prompting import get_detection_template, get_simple_detection_template
from config import DetectionScorerConfig

CONFIG = DetectionScorerConfig()

class DetectionScorer:

    def __init__(
        self,
        model,
        state,
    ):
        self.model = model
        self.state = state

    def get_mixed_examples_list(self, l_ctx, r_ctx):

        n_examples = CONFIG.n_real * CONFIG.n_batches

        assert (len(CONFIG.real_ids) == CONFIG.n_real)

        true_examples_list = [i[0] for i in self.state.examples[:n_examples]]

        mixed_examples_list = self.generate_random_examples(l_ctx, r_ctx)

        for batch in range(CONFIG.n_batches):
            for i in range(CONFIG.n_real):
                mixed_examples_list[batch*CONFIG.batch_size + CONFIG.real_ids[i]] = true_examples_list[batch*CONFIG.n_real + i]

        mixed_examples_list = self.model.tokenizer.batch_decode(mixed_examples_list)
        mixed_examples_list = [example.replace("{", "{{").replace("}", "}}") for example in mixed_examples_list]

        return mixed_examples_list


    def generate_random_examples(self, l_ctx, r_ctx):
        mixed_examples_list = []

        while len(mixed_examples_list) < CONFIG.n_batches*CONFIG.batch_size:
            # Choose a random batch and sequence position
            batch = t.randint(0, self.state.act_cache.size(0), [1]).squeeze()
            pos = t.randint(l_ctx, self.state.tok_cache.size(1) - r_ctx, [1]).squeeze()

            # Extract the tokens and activations for the fake example
            fake_example_toks = self.state.tok_cache[batch, pos - l_ctx : pos + r_ctx + 1]
            fake_example_acts = self.state.act_cache[batch, pos - l_ctx : pos + r_ctx + 1, self.state.feature_id]

            # Check the fake example does not activate the real one
            if sum(fake_example_acts) < 0.01:
                # Append the fake example to the list
                mixed_examples_list.append(fake_example_toks)

        return mixed_examples_list


    def score(self, explanation_list, l_ctx, r_ctx):
        mixed_examples_list = self.get_mixed_examples_list(l_ctx, r_ctx)
        score_list = []

        log(self, "section", "Running detection scoring.")
        log(self, "user", get_simple_detection_template("<EXPLANATION>", "<MIXED EXAMPLES>"))

        for explanation in tqdm(explanation_list):
            summed_detection_rate = 0.0
            summed_false_pos_rate = 0.0

            log(self, "system", f"Running on explanation: {explanation}")

            for b in range(CONFIG.n_batches):
                log(self, "system", f"Processing batch {b+1} of {CONFIG.n_batches}")

                mixed_examples = mixed_examples_list[b*2*CONFIG.n_real : (b+1)*2*CONFIG.n_real]
                mixed_examples_str = ''
                for i in range(len(mixed_examples)):
                    mixed_examples_str += f'Example {i+1}: {mixed_examples[i]}\n'

                detection_rate, false_pos_rate = self.query(explanation, mixed_examples_str)

                summed_detection_rate += detection_rate
                summed_false_pos_rate += false_pos_rate

            detection_rate = summed_detection_rate /  CONFIG.n_batches
            false_pos_rate = summed_false_pos_rate /  CONFIG.n_batches
            score_list.append((detection_rate, false_pos_rate))

        return score_list


    def query(self, explanation, mixed_examples_str):
  
        generation_kwargs = {
            "max_tokens":CONFIG.max_tokens,
            "temperature":CONFIG.temperature
        }
        output = gen(
            get_detection_template(mixed_examples_str, explanation), 
            generation_kwargs=generation_kwargs
        )
 
        log(self, "assistant", "".join(output))

        output_str = "".join(output)

        nums = self.extract_numbers_from_string(output_str)

        detection_rate = sum([num-1 in CONFIG.real_ids for num in nums]) / CONFIG.n_real
        false_pos_rate = sum([num-1 not in CONFIG.real_ids for num in nums]) / CONFIG.n_real

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


    