from abc import ABC, abstractmethod
import openai
import os
import replicate

class Client(ABC):
    def __init__(self, model: str):
        self.model = model

    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass

class OpenAI(Client):
    def __init__(self, model: str, api_key: str):
        super().__init__(model)
        self.client = openai.OpenAI(api_key=api_key)
    
    def generate(self, prompt: str) -> str:
        return self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": prompt}]
        ).choices[0].message.content
    
class Replicate(Client):
    def __init__(self, model: str, api_key: str):
        super().__init__(model)
        os.environ["REPLICATE_API_TOKEN"] = api_key
    
    def generate(self, prompt: str) -> str:

        prompt = {
            "prompt":"",
            "prompt_template": prompt,
        }
            
        return replicate.run(
            self.model,
            input=prompt
        )
    

def get_client(provider: str, api_key: str):
    if provider is None or api_key is None:
        return None 

    if provider == "openai":
        model = "gpt-4o"
        return OpenAI(model, api_key)
    
    if provider == "replicate":
        model = "meta/meta-llama-3-70b-instruct"
        return Replicate(model, api_key)
