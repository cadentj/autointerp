from dataclasses import dataclass
from prompting import get_gen_scorer_template, get_simple_gen_scorer_template
from utils import gen

@dataclass
class GenerationScorerConfig:
    n_examples : int = 10
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

        self.state.history.append({"role":"assistant","message": "".join(output)})

        s = ''
        for i in output:
            s += i

        examples = [e.strip("Example ").strip("1234567890").strip(": ") for e in s.split('\n') if e.startswith("Example")]
        examples = [e for e in examples if len(e) > 0]

        return examples


    def score(self, explanation_list, sae):
        scores_list = []

        self.state.history.append({"role":"section","message":f"Running generation scoring."})
        self.state.history.append({"role":"user","message":get_simple_gen_scorer_template("<EXPLANATION>", self.cfg.n_examples)})

        for explanation in explanation_list:
            self.state.history.append({"role":"system","message":f"Running on explanation: {explanation}"})

            examples = self.get_llm_examples(explanation)

            with self.model.trace(examples):
                activations = self.model.transformer.h[self.state.layer].input[0][0]

                middle = sae(activations)

                feature_acts = middle[1]
                feature_acts.save()

            score_batch = feature_acts[:,:,self.state.feature_id].max(dim=1)[0]

            scores_list.append( score_batch.mean() )

            self.state.history.append({"role":"system","message":f"Score: {scores_list[-1]}"})

            del feature_acts

        return scores_list