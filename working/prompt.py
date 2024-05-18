from abc import ABC, abstractmethod
from . import CONFIG
import openai

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

def get_client(provider: str, api_key: str):
    if provider == "openai":
        return OpenAI(provider, api_key)
    
def create(prompt: str):
    if CONFIG.API.PROVIDER == "openai":
        return OpenAI(CONFIG.API.PROVIDER, CONFIG.API.APIKEY).generate(prompt)
