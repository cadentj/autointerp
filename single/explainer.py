from tqdm import tqdm

from utils import gen, log
from prompting import get_explainer_template, get_simple_explainer_template
from config import ExplainerConfig

class Explainer:

    def __init__(
        self, 
        model,
        state, 
        cfg: ExplainerConfig
    ):  

        self.state = state
        self.model = model
        self.cfg = cfg

        # n_exampels should be less than or equal to the number that we cached
        assert(self.cfg.batch_size * self.cfg.n_batches <= len(self.state.examples))

        self.n_examples = self.cfg.batch_size * self.cfg.n_batches


    def fix(self, string):
        string = string.replace(self.cfg.r+self.cfg.l, "")
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
                    delimited_string += self.cfg.l + self.model.tokenizer.decode(example_toks[pos]) + self.cfg.r
                else:
                    delimited_string += self.model.tokenizer.decode(example_toks[pos])

            delimited_string = self.fix(delimited_string)

            top_examples_list.append(delimited_string)

        return top_examples_list


    def explain(self):
        top_examples_list = self.prepare_top_examples()

        explanation_list = []
        
        log(self.state, "section", "Running explainer.")
        log(self.state, "user", get_simple_explainer_template("<EXAMPLES>"))

        for batch in tqdm(range(self.cfg.n_batches), desc="Processing batches"):

            log(self.state, "system", f"Processing batch {batch+1} of {self.cfg.n_batches}")

            # get batch of top examples, and convert to string
            examples_list = top_examples_list[batch*self.cfg.batch_size : (batch+1)*self.cfg.batch_size]
            examples_str = ""

            for i in range(len(examples_list)):
                examples_str += "Example " + str(i+1) + ": " + examples_list[i] + "\n"

            for trial in tqdm(range(self.cfg.runs_per_batch), desc="Running queries", leave=False):

                log(self.state, "system", f"Query {trial+1} of {self.cfg.runs_per_batch}")
                log(self.state, "system", f"Running on examples:\n{examples_str}")

                two_explanations = self.query(examples_str)
                
                explanation_list += two_explanations

        return explanation_list


    def query(self, examples_str):

        generation_kwargs = {
            "max_tokens":self.cfg.max_tokens,
            "temperature":self.cfg.temperature
        }
        
        two_explanations, output = gen(
            get_explainer_template(examples_str), 
            postprocess = self.split_explanations,
            generation_kwargs=generation_kwargs
        )

        log(self.state, "assistant", output)

        return two_explanations
    
    def split_explanations(self, s):
        two_explanations = s.split("Step 4")[-1].split("1")[-1].split("2")  # this is janky as fuck, change this
        two_explanations = [e.strip(".): ") for e in two_explanations]

        return two_explanations