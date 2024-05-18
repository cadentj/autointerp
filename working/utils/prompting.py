from abc import ABC, abstractmethod
import openai

class Client(ABC):
    def __init__(self, model: str):
        self.model = model

    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass

    # @abstractmethod
    # def parse_args(self, args: dict) -> dict:
    #     pass


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
    if provider is None or api_key is None:
        return None 

    if provider == "openai":
        model = "gpt-4o"
        return OpenAI(model, api_key)
