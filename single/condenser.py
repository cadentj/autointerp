from utils import gen
from prompting import get_condenser_template

def condense(explanation_list, return_output=False):
    prompt = {
        "prompt":  '',
        "prompt_template": get_condenser_template(explanation_list),
        "max_tokens" : 2000,
        "temperature" : 0.0
    }
    output = gen(prompt)


    output_str = ''
    for i in output:
      output_str += i

    exps =  [exp.strip("Feature").strip(" ").strip("12345678").strip(": ") for exp in output_str.split('\n') if exp.startswith("Feature")]

    if return_output:
        return exps, output
        
    return exps