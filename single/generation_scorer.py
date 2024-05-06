from dataclasses import dataclass
from prompting import get_gen_scorer_template
from utils import gen

@dataclass
class GenerationScorerConfig:
    n_examples : int = 10
    point : str = "resid_pre"
    layer : int = 10
    verbose : bool = True

class GenerationScorer:

    def __init__(
        self,
        model,
        state,
        cfg: GenerationScorerConfig = None
    ):
        self.model = model
        self.state = state
    
        if cfg is None:
            cfg = GenerationScorerConfig()
        self.cfg = cfg

    def get_llm_examples(self, explanation):

        prompt = {
            "prompt":  '',
            "prompt_template": get_gen_scorer_template(explanation, self.cfg.n_examples),
            "max_tokens" : 2000,
            "temperature" : 0.5,
            "frequency_penalty" : 0.0,
            "presence_penalty" : 0.0,
            "top_p" : 1.0,

        }
        output = gen(prompt)

        s = ''
        for i in output:
            s += i

        examples = [e.strip("Example ").strip("1234567890").strip(": ") for e in s.split('\n') if e.startswith("Example")]
        examples = [e for e in examples if len(e) > 0]

        return examples


    def score(self, feature_id, explanation_list):
        scores_list = []
        
        for explanation in explanation_list:
            examples = self.get_llm_examples(explanation)

            with self.model.trace(examples):
                activations = self.model.transformer.h[self.state.layer].input[0][0]

                middle = self.sae_list[self.state.layer](activations)

                feature_acts = middle[1]
                feature_acts.save()

            score_batch = feature_acts[:,:,feature_id].max(dim=1)[0]

            scores_list.append( score_batch.mean() )

            del feature_acts

        return scores_list