from utils import gen
from prompting import get_condenser_template
from config import CondenserConfig

CONFIG = CondenserConfig()

def condense(explanation_list, return_output=False):
    generation_kwargs = {
        "max_tokens" : CONFIG.max_tokens,
        "temperature" : CONFIG.temperature
    }
    
    output = gen(
        get_condenser_template(explanation_list),
        generation_kwargs=generation_kwargs
    )

    output_str = ''
    for i in output:
      output_str += i

    exps =  [exp.strip("Feature").strip(" ").strip("12345678").strip(": ") for exp in output_str.split('\n') if exp.startswith("Feature")]

    if return_output:
        return exps, output
        
    return exps