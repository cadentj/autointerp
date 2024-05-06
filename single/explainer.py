from dataclasses import dataclass
from utils import topk
from tqdm import tqdm
from utils import gen
from prompting import get_explainer_template

@dataclass
class ExplainerConfig:

    max_tokens : int = 2000
    temperature : float = 0.5

    batch_size: int = 10
    n_batches : int = 2
    runs_per_batch: int = 1

    left_ctx: int = 15
    right_ctx: int = 2
    activation_threshold: float = 0.4

# delimiters
l = '<<'
r = '>>'

class Explainer:

    def __init__(
        self, 
        model,
        state, 
        cfg: ExplainerConfig = None
    ):  
        if cfg is None:
            cfg = ExplainerConfig()

        self.cfg = cfg

        self.state = state
        self.model = model

        # n_exampels should be less than or equal to the number that we cached
        assert(cfg.batch_size * cfg.n_batches <= len(self.state.examples))

        self.n_examples = cfg.batch_size * cfg.n_batches

    def fix(self, string):
        string = string.replace(r+l, "")
        string = string.replace("{", "{{").replace("}", "}}")
        return string

    def prepare_top_examples(self):

        top_examples_list = []

        examples = self.state.examples[:self.n_examples]

        for example in examples:

            example_toks, example_acts = example

            delimited_string = ''
            for pos in range(example_toks.size(0)):
                if example_acts[pos] > (self.cfg.activation_threshold * self.state.max_act):
                    delimited_string += l + self.model.tokenizer.decode(example_toks[pos]) + r
                else:
                    delimited_string += self.model.tokenizer.decode(example_toks[pos])

            delimited_string = self.fix(delimited_string)

            top_examples_list.append(delimited_string)

        return top_examples_list

    def explain(self):
        top_examples_list = self.prepare_top_examples()

        explanation_list = []

        for batch in tqdm(range(self.cfg.n_batches)):

            # get batch of top examples, and convert to string
            examples_list = top_examples_list[batch*self.cfg.batch_size : (batch+1)*self.cfg.batch_size]
            examples_str = ""

            for i in range(len(examples_list)):
                examples_str += "Example " + str(i+1) + ": " + examples_list[i] + "\n"

                for _ in range(self.cfg.runs_per_batch):

                    two_explanations = self.query(examples_str)
                    
                    explanation_list.append(two_explanations)

        return explanation_list

    def query(self, examples_str):
        prompt = {
            "prompt": examples_str,
            "prompt_template": get_explainer_template(examples_str),
            "max_tokens" : self.cfg.max_tokens,
            "temperature" : self.cfg.temperature
        }

        output = gen(prompt)

        output_str = ''
        for i in output:
            output_str += i

        two_explanations = output_str.split("Step 4")[-1].split("1")[-1].split("2")  # this is janky as fuck, change this
        two_explanations = [e.strip(".): ") for e in two_explanations]

        return two_explanations