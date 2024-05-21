from abc import ABC, abstractmethod
import openai
from groq import Groq as GroqClient
import os
import replicate
from transformers import AutoTokenizer

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
            messages=prompt,
            temperature=0.8
        ).choices[0].message.content


class Groq(Client):
    def __init__(self, model: str, api_key: str):
        super().__init__(model)
        self.client = GroqClient(api_key=api_key)
    
    def generate(self, prompt: str) -> str:
        return self.client.chat.completions.create(
            model=self.model,
            messages=prompt
        ).choices[0].message.content
    

class Replicate(Client):
    def __init__(self, model: str, api_key: str, tokenizer):
        super().__init__(model)
        os.environ["REPLICATE_API_TOKEN"] = api_key
        self.tokenizer = tokenizer
    
    def generate(self, prompt) -> str:
        if prompt is not type(str):
            prompt = self.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)

        prompt = {
            "prompt":"",
            "prompt_template": prompt,
            "temperature": 0.8,
        }
            
        output = replicate.run(
            self.model,
            input=prompt
        )

        return "".join(output)
    

def get_client(provider: str, api_key: str):
    if provider is None or api_key is None:
        return None 

    if provider == "openai":
        model = "gpt-4-turbo"
        return OpenAI(model, api_key)
    
    if provider == "replicate":
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-70B-Instruct")
        model = "meta/meta-llama-3-8b-instruct"
        return Replicate(model, api_key, tokenizer)

    if provider == "groq":
        model = "llama3-8b-8192"
        return Groq(model, api_key)
