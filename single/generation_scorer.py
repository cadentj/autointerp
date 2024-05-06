from utils import gen, log
from prompting import get_gen_scorer_template, get_simple_gen_scorer_template
from config import GenerationScorerConfig

CONFIG = GenerationScorerConfig()

class GenerationScorer:

    def __init__(
        self,
        model,
        state,
    ):
        self.model = model
        self.state = state

    def get_llm_examples(self, explanation):

        generation_kwargs = {
            "max_tokens" : 2000,
            "temperature" : 0.5,
            "frequency_penalty" : 0.0,
            "presence_penalty" : 0.0,
            "top_p" : 1.0,
        }

        output = gen(
            get_gen_scorer_template(explanation, CONFIG.n_examples), 
            generation_kwargs=generation_kwargs
        )

        log(self, "assistant", "".join(output))

        output_str = "".join(output)

        examples = [e.strip("Example ").strip("1234567890").strip(": ") for e in output_str.split('\n') if e.startswith("Example")]
        examples = [e for e in examples if len(e) > 0]

        return examples


    def score(self, explanation_list, sae):
        scores_list = []

        log(self, "section", "Running generation scoring.")
        log(self, "user", get_simple_gen_scorer_template("<EXPLANATION>", CONFIG.n_examples))

        for explanation in explanation_list:
            log(self, "system", f"Running on explanation: {explanation}")

            examples = self.get_llm_examples(explanation)

            with self.model.trace(examples):
                activations = self.model.transformer.h[self.state.layer].input[0][0]

                middle = sae(activations)

                feature_acts = middle[1]
                feature_acts.save()

            score_batch = feature_acts[:,:,self.state.feature_id].max(dim=1)[0]

            scores_list.append( score_batch.mean() )

            log(self, "system", f"Score: {scores_list[-1]}")

            del feature_acts

        return scores_list