GSCORER_SYSTEM = """You will be presented with a description of a certain feature of text, e.g. "male names" or "text with positive sentiment".

Your task is to come up with a requested number of examples of short text samples that contain the exact feature that was described to you.

Guidelines:
- You must pay extremely careful attention to the exact details of the feature being decribed.
- Every single example you write must possess the feature.
- Try to keep the examples varied.
- Unless the feature description explicitly refers to position within a sentence, you should make sure to vary the position in the sentence at which the feature appears. E.g. the word/token at which the feature appears should not always be at the start of the sentence.
- IMPORTANT: If the feature explanation involves some broader context of the text, you must establish the context at the start of each example. By the time the feature appears in the example, the context must already have been established.

Think carefully first. Then output the list of examples, one per line, starting with "[EXAMPLE]: " and so on."""

EXAMPLE_1 = """Generate 10 examples. Description of text feature: male names in text to do with sports."""

RESPONSE_1 = """Here are 10 examples of short text examples containing male names in the context of sports.

Since the feature involves a general context (sports), I'll make sure to establish that context at the start of each example.

I'll keep the examples varied, with the feature appearing at different positions in the sentence.

Example 1: The olympic gold medal went to Bob.
Example 2: Arsenal won the league that year. Gary Lineker was angry about it.
Example 3: The next tennis match will be between Andy Murray and Roger Federer.
Example 4: The greatest NBA player of all time was Michael Jordan, no debate.
Example 5: The Warriors beat the Nuggets 104-102 in overtime, with a clutch 3-pointer from Stephen Curry yet again.
Example 6: When you think of hockey, you think of Wayne Gretzky.
Example 7: WHY DO LIVERPOOL KEEP LOSING. WHAT HAPPENED TO SCORING GOALS?? FUCK JURGEN KLOPP. FUCK LIVERPOOL.
Example 8: Yet another superbowl for the best QB in the NFL: Patrick Mahomes.
Example 9: He's boxing like Mike Tyson in his prime.
Example 10: Top scorer of all time: Lionel Messi."""

START = """Generate {n_examples} examples. Description of text feature: {description}"""

gen_scorer_prompt = lambda n_examples, description : [
    {
        "role" : "system",
        "content" : GSCORER_SYSTEM
    },
    {
        "role" : "user",
        "content" : EXAMPLE_1
    },
    {
        "role" : "assistant",
        "content" : RESPONSE_1
    },
    {
        "role" : "user",
        "content" : START.format(n_examples, description)
    }
]