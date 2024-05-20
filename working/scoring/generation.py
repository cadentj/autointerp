from ..utils.api import Client
from ..prompting.scoring_prompts import gen_scorer_prompt

class GenerationScorer():  
    
    def __init__(
        self,
        client: Client, 
    ):
        self.client = client

    def generate_examples(
        self,
        n_examples: int,
        explanation: str,
        generation_args: dict,
    ):
        prompt = gen_scorer_prompt(n_examples, explanation)
        return self.client.generate(
            prompt
        )
    
    def save(
        self,
        id,
        explanation,
        examples
    ):
        import json 

        data = {
            "explanation" : explanation,
            "examples" : examples
        }

        with open(f"{id}.json", "w") as f:
            json.dump(data, f)
    
    def score(
        self,
        n_examples,
        explanation,
        id
    ):

        examples = self.generate_examples(
            n_examples,
            explanation,
            {}
        )

        self.save(
            id,
            explanation,
            examples
        )

        return examples  
        
    