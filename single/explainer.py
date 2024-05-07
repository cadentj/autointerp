from tqdm import tqdm

from utils import gen, log
from prompting import get_explainer_template, get_simple_explainer_template
from config import ExplainerConfig

CONFIG = ExplainerConfig()

class Explainer:

    def __init__(
        self, 
        model,
        state, 
    ):  

        self.state = state
        self.model = model

        # n_exampels should be less than or equal to the number that we cached
        assert(CONFIG.batch_size * CONFIG.n_batches <= len(self.state.examples))

        self.n_examples = CONFIG.batch_size * CONFIG.n_batches


    def fix(self, string):
        string = string.replace(CONFIG.r+CONFIG.l, "")
        string = string.replace("{", "{{").replace("}", "}}")
        return string


    def prepare_top_examples(self):

        top_examples_list = []

        examples = self.state.examples[:self.n_examples]

        for example in examples:

            example_toks, example_acts = example

            delimited_string = ''
            for pos in range(example_toks.size(0)):
                if example_acts[pos] > (CONFIG.activation_threshold * self.state.max_act):
                    delimited_string += CONFIG.l + self.model.tokenizer.decode(example_toks[pos]) + CONFIG.r
                else:
                    delimited_string += self.model.tokenizer.decode(example_toks[pos])

            delimited_string = self.fix(delimited_string)

            top_examples_list.append(delimited_string)

        return top_examples_list


    def explain(self):
        top_examples_list = self.prepare_top_examples()

        explanation_list = []
        
        log(self, "section", "Running explainer.")
        log(self, "user", get_simple_explainer_template("<EXAMPLES>"))

        for batch in tqdm(range(CONFIG.n_batches), desc="Processing batches"):

            log(self, "system", f"Processing batch {batch+1} of {CONFIG.n_batches}")

            # get batch of top examples, and convert to string
            examples_list = top_examples_list[batch*CONFIG.batch_size : (batch+1)*CONFIG.batch_size]
            examples_str = ""

            for i in range(len(examples_list)):
                examples_str += "Example " + str(i+1) + ": " + examples_list[i] + "\n"

            for trial in tqdm(range(CONFIG.runs_per_batch), desc="Running queries", leave=False):

                log(self, "system", f"Query {trial+1} of {CONFIG.runs_per_batch}")
                log(self, "system", f"Running on examples:\n{examples_str}")

                two_explanations = self.query(examples_str)
                
                explanation_list.append(two_explanations)

        return explanation_list


    def query(self, examples_str):

        generation_kwargs = {
            "max_tokens":CONFIG.max_tokens,
            "temperature":CONFIG.temperature
        }
        
        two_explanations, output = gen(
            get_explainer_template(examples_str), 
            postprocess = self.split_explanations,
            generation_kwargs=generation_kwargs
        )

        log(self, "assistant", output)

        return two_explanations
    
    def split_explanations(self, s):
        two_explanations = s.split("Step 4")[-1].split("1")[-1].split("2")  # this is janky as fuck, change this
        two_explanations = [e.strip(".): ") for e in two_explanations]

        return two_explanations