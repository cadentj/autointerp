from typing import List

class Quote:
    def __init__(
        self,
        examples = List[str]
    ):
        self.text = "\n".join(examples)

    def verify_quote(self, quote: str) -> bool:
        return quote in self.text
    
    def __call__(self, quote: str) -> bool:
        if self.verify_quote(quote):
            return f"<verified>{quote}</verified>"
        else:
            return f"<unverified>{quote}</unverified>"