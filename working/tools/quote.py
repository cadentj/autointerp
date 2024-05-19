from typing import List
import re

class Quote:
    def __init__(
        self,
        examples = List[str]
    ):
        self.text = "\n".join(examples)

    def verify_quote(self, quote: str) -> bool:
        return quote in self.text
    
    def extract_quotes(self, paragraph):
        pattern = r'\[QUOTE\]: (.+)'
        matches = re.findall(pattern, paragraph)
        return matches
    
    def replace_quotes(self, paragraph, quotes):
        for quote in quotes:
            if self.verify_quote(quote):
                paragraph = paragraph.replace(quote, f"<verified>{quote}</verified>")
            else:
                paragraph = paragraph.replace(quote, f"<unverified>{quote}</unverified>")
        return paragraph
    
    def __call__(self, argument: str): 
        quotes = self.extract_quotes(argument)
        verified_argument = self.replace_quotes(argument, quotes)
        return verified_argument