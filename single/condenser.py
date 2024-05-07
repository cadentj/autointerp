from utils import gen
from prompting import get_condenser_template
from config import CondenserConfig

CONFIG = CondenserConfig()

def condense(explanation_list, return_output=True):
    generation_kwargs = {
        "max_tokens" : CONFIG.max_tokens,
        "temperature" : CONFIG.temperature
    }
    
    exps, output_str = gen(
        get_condenser_template(explanation_list),
        postprocess=strip_numbering,
        generation_kwargs=generation_kwargs
    )

    if return_output:
        return exps, output_str
        
    return exps

def strip_numbering(s):
    return [exp.strip("Feature").strip(" ").strip("12345678").strip(": ") for exp in s.split('\n') if exp.startswith("Feature")]