### SYSTEM PROMPT ###

SYSTEM = """You are a meticulous AI researcher conducting an important investigation into patterns found in language. Your task is to analyze text and return a score as to how well that text agrees with a query.

Guidelines:

You will be given a list of text examples on which special words are selected and between delimiters like <<this>>. If a sequence of consecutive tokens all are important, the entire sequence of tokens will be contained between delimiters <<just like this>>. How important each token is for the behavior is listed after each example in parentheses.

- Produce a single final score. This score should reflect how relevant the provided information is to the query.
- If the examples are uninformative, you can ignore them.
- Do not mention the marker tokens (<< >>) in your explanation. They are not part of the actual text, only used to highlight the important tokens.
- The last line of your response must be the formatted score, an integer between 0 and 3 inclusive, using [SCORE]:.

{prompt}
"""


COT = """
To better find the explanation for the language patterns go through the following stages:

1. Find the special words that are selected in the examples and list a couple of them. Search for patterns in these words, if there are any. Don't list more than 5 words.

2. Write down general shared features of the text examples. This could be related to the full sentence or to the words surrounding the marked words.

3. Formulate an hypothesis and write down your final score using [SCORE]:.

"""



### EXAMPLE 1 ###

EXAMPLE_1 = """QUERY: Statements about barnyard animals.

Example 1:  and he was <<over the moon>> to find
Activations: ("over", 5), (" the", 6), (" moon", 9)
Example 2:  we'll be laughing <<till the cows come home>>! Pro
Activations: ("till", 5), (" the", 5), (" cows", 8), (" come", 8), (" home", 8)
Example 3:  thought Scotland was boring, but really there's more <<than meets the eye>>! I'd
Activations: ("than", 5), (" meets", 7), (" the", 6), (" eye", 8)
"""


EXAMPLE_1_COT_ACTIVATION_RESPONSE = """
ACTIVATING TOKENS: "over the moon", "than meets the eye".
SURROUNDING TOKENS: No interesting patterns.

Step 1.
- The activating tokens are all parts of common idioms.
- The surrounding tokens have nothing in common.

Step 2.
- The examples contain common idioms.
- In some examples, the activating tokens are followed by an exclamation mark.

Step 3.
- The activation values are the highest for the more common idioms in examples 1 and 3.
- The text examples all convey positive sentiment.
- The query is for statements about barnyard animals, but the examples are about positive, common idioms.
"""

EXAMPLE_1_EXPLANATION = """
[SCORE]: 0
"""

### EXAMPLE 2 ###

EXAMPLE_2 = """QUERY: Comparison features, descriptions that differentiate between multiple objects.

Example 1:  a river is wide but the ocean is wid<<er>>. The ocean
Activations: ("er", 8)
Example 2:  every year you get tall<<er>>," she
Activations: ("er", 2)
Example 3:  the hole was small<<er>> but deep<<er>> than the
Activations: ("er", 9), ("er", 9)
"""

EXAMPLE_2_COT_ACTIVATION_RESPONSE = """
ACTIVATING TOKENS: "er", "er", "er".
SURROUNDING TOKENS: "wid", "tall", "small", "deep".

Step 1.
- The activating tokens are mostly "er".
- The surrounding tokens are mostly adjectives, or parts of adjectives, describing size.
- The neuron seems to activate on, or near, the tokens in comparative adjectives describing size.

Step 2.
- In each example, the activating token appeared at the end of a comparative adjective.
- The comparative adjectives ("wider", "taller", "smaller", "deeper") all describe size.

Step 3.
- Example 2 has a lower activation value. It doesn't compare sizes as directly as the other examples.
- The query is for comparison features, and the feature activates on words that are part of a comparative adjective.
"""


EXAMPLE_2_EXPLANATION = """
[SCORE]: 2
"""

### EXAMPLE 3 ###

EXAMPLE_3 = """QUERY: Objects which represent a distinct space, for example, containers.

Example 1:  something happening inside my <<house>>", he
Activations: ("house", 7)
Example 2:  presumably was always contained in <<a box>>", according
Activations: ("a", 5), ("box", 9)
Example 3:  people were coming into the <<smoking area>>".

However he
Activations: ("smoking", 2), ("area", 4)
Example 4:  Patrick: "why are you getting in the << way?>>" Later,
Activations: ("way", 4), ("?", 2)
"""

EXAMPLE_3_COT_ACTIVATION_RESPONSE = """
ACTIVATING TOKENS: "house", "a box", "smoking area", " way?".
SURROUNDING TOKENS: No interesting patterns.

Step 1.
- The activating tokens are all things that one can be in.
- The surrounding tokens have nothing in common.

Step 2.
- The examples involve being inside something, sometimes figuratively.
- The activating token is a thing which something else is inside of.

STEP 3.
- The activation values are highest for the examples where the token is a distinctive object or space.
- The activating token is followed by a quotation mark, suggesting it occurs within speech.
- The feature activates on containers like "house" and "box", and the query is directly related.
"""

EXAMPLE_3_EXPLANATION = """
[SCORE]: 3
"""

EXAMPLES = [
    (EXAMPLE_1, EXAMPLE_1_COT_ACTIVATION_RESPONSE, EXAMPLE_1_EXPLANATION),
    (EXAMPLE_2, EXAMPLE_2_COT_ACTIVATION_RESPONSE, EXAMPLE_2_EXPLANATION),
    (EXAMPLE_3, EXAMPLE_3_COT_ACTIVATION_RESPONSE, EXAMPLE_3_EXPLANATION),
]

def system(use_cot: bool = False):
    prompt = ""

    if use_cot:
        prompt += COT

    return [
        {
            "role": "system",
            "content": SYSTEM.format(prompt=prompt),
        }
    ]

def build_examples(
    use_cot: bool = False,
):
    examples = []

    for example, cot_response, explanation in EXAMPLES:
        response = cot_response + explanation if use_cot else explanation

        messages = [
            {
                "role": "user",
                "content": example,
            },
            {
                "role": "assistant",
                "content": response,
            },
        ]

        examples.extend(messages)

    return examples


def build_prompt(
    examples: str,
    query: str,
    use_cot: bool = False,
):
    messages = system(
        use_cot=use_cot,
    )

    few_shot_examples = build_examples(
        use_cot=use_cot,
    )

    messages.extend(few_shot_examples)

    user_start = f"QUERY: {query}\n\n{examples}\n"

    messages.append(
        {
            "role": "user",
            "content": user_start,
        }
    )

    return messages
