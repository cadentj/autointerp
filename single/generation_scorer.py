from utils import gen, log
from prompting import get_gen_scorer_template, get_simple_gen_scorer_template
from config import GenerationScorerConfig

class GenerationScorer:

    def __init__(
        self,
        model,
        state,
        cfg: GenerationScorerConfig
    ):
        self.model = model
        self.state = state
        self.cfg = cfg

    def get_llm_examples(self, explanation):

        generation_kwargs = {
            "max_tokens" : 2000,
            "temperature" : self.cfg.temperature,
            "frequency_penalty" : 0.0,
            "presence_penalty" : 0.0,
            "top_p" : 1.0,
        }

        examples, output_str = gen(
            get_gen_scorer_template(explanation, self.cfg.n_examples), 
            postprocess=self.strip_numbering,
            generation_kwargs=generation_kwargs
        )

        log(self.state, "assistant", output_str)

        return examples


    def strip_numbering(self, s):
        examples = [e.strip("Example ").strip("1234567890").strip(": ") for e in s.split('\n') if e.startswith("Example")]
        return [e for e in examples if len(e) > 0]
    

    def score(self, explanation_list, sae):
        scores_list = []

        log(self.state, "section", "Running generation scoring.")
        log(self.state, "user", get_simple_gen_scorer_template("<EXPLANATION>", self.cfg.n_examples))

        for explanation in explanation_list:
            log(self.state, "system", f"Running on explanation: {explanation}")

            examples = self.get_llm_examples(explanation)

            with self.model.trace(examples):
                activations = self.model.transformer.h[self.state.layer].input[0][0]

                middle = sae(activations)

                feature_acts = middle[1][:,:,self.state.feature_id]
                feature_acts[:,0] = 0
                feature_acts.save()
                
            score_batch = feature_acts.max(dim=1)[0]

            scores_list.append( score_batch.mean() )

            log(self.state, "system", f"Score: {scores_list[-1]}")

            del feature_acts

        return scores_list