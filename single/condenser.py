from utils import gen
from prompting import get_condenser_template
from config import CondenserConfig

class Condenser:
    def __init__(self, cfg: CondenserConfig):
        self.cfg = cfg

    def condense(self, explanation_list, return_output=True):
        generation_kwargs = {
            "max_tokens" : self.cfg.max_tokens,
            "temperature" : self.cfg.temperature
        }
        
        exps, output_str = gen(
            get_condenser_template(explanation_list),
            postprocess=self.strip_numbering,
            generation_kwargs=generation_kwargs
        )

        if return_output:
            return exps, output_str
            
        return exps

    def strip_numbering(self, s):
        return [exp.strip("Feature").strip(" ").strip("12345678").strip(": ") for exp in s.split('\n') if exp.startswith("Feature")]