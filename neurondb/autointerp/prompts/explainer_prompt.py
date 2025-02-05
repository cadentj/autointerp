### SYSTEM PROMPT ###

SYSTEM = """You are a meticulous AI researcher conducting an important investigation into patterns found in language. Your task is to analyze text and provide an explanation that thoroughly encapsulates possible patterns found in it.
Guidelines:

You will be given a list of text examples on which special words are selected and between delimiters like <<this>>. If a sequence of consecutive tokens all are important, the entire sequence of tokens will be contained between delimiters <<just like this>>. How important each token is for the behavior is listed after each example in parentheses. After each example is a set of activations, indicating the importance of each token in the example.

- Produce a concise, one or two sentence final description. Simply describe the text features that are common in the examples, and what patterns you found.
- If the examples are uninformative, you don't need to mention them. Don't focus on giving examples of important tokens, but try to summarize the patterns found in the examples.
- Do not mention the marker tokens (<< >>) in your explanation.
- Examples without marker tokens are typically uninformative. Their activations were too low to be marked.
- Assume someone reading your explanation does not have access to the examples you saw. That means you should not point to specific examples in your explanation.
- The last line of your response must be the formatted explanation, preceded by [EXPLANATION]:.{prompt}"""


COT = """To better find the explanation for the language patterns go through the following stages:

1. Find the special words that are selected in the examples and list a couple of them. Search for patterns in these words, if there are any. Don't list more than 5 words.
2. Write down general shared features of the text examples. This could be related to the full sentence or to the words surrounding the marked words.
3. Formulate an hypothesis and write down the final explanation using [EXPLANATION]:."""



### EXAMPLE 1 ###

EXAMPLE_1 = """Example 1:  and he was <<over the moon>> to find
Activations: ("over", 5), (" the", 6), (" moon", 9)
Example 2:  we'll be laughing <<till the cows come home>>! Pro
Activations: ("till", 5), (" the", 5), (" cows", 8), (" come", 8), (" home", 8)
Example 3:  thought Scotland was boring, but really there's more <<than meets the eye>>! I'd
Activations: ("than", 5), (" meets", 7), (" the", 6), (" eye", 8)"""


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
"""

EXAMPLE_1_EXPLANATION = """[EXPLANATION]: Common idioms in text conveying positive sentiment."""

### EXAMPLE 2 ###

EXAMPLE_2 = """Example 1:  a river is wide but the ocean is wid<<er>>. The ocean
Activations: ("er", 8)
Example 2:  every year you get tall<<er>>," she
Activations: ("er", 2)
Example 3:  the hole was small<<er>> but deep<<er>> than the
Activations: ("er", 9), ("er", 9)"""

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
"""

EXAMPLE_2_EXPLANATION = """[EXPLANATION]: The token "er" at the end of a comparative adjective describing size."""

### EXAMPLE 3 ###

EXAMPLE_3 = """Example 1:  something happening inside my <<house>>", he
Activations: ("house", 7)
Example 2:  presumably was always contained in <<a box>>", according
Activations: ("a", 5), ("box", 9)
Example 3:  people were coming into the <<smoking area>>".

However he
Activations: ("smoking", 2), ("area", 4)
Example 4:  Patrick: "why are you getting in the << way?>>" Later,
Activations: ("way", 4), ("?", 2)"""

EXAMPLE_3_COT_ACTIVATION_RESPONSE = """ACTIVATING TOKENS: "house", "a box", "smoking area", " way?".
SURROUNDING TOKENS: No interesting patterns.

Step 1.
- The activating tokens are all things that one can be in.
- The surrounding tokens have nothing in common.

Step 2.
- The examples involve being inside something, sometimes figuratively.
- The activating token is a thing which something else is inside of.

Step 3.
- The activation values are highest for the examples where the token is a distinctive object or space.
- The activating token is followed by a quotation mark, suggesting it occurs within speech.
"""

EXAMPLE_3_EXPLANATION = """[EXPLANATION]: Nouns representing distinct objects that contain something, sometimes preceding a quotation mark."""

EXAMPLES = [
    (EXAMPLE_1, EXAMPLE_1_COT_ACTIVATION_RESPONSE, EXAMPLE_1_EXPLANATION),
    (EXAMPLE_2, EXAMPLE_2_COT_ACTIVATION_RESPONSE, EXAMPLE_2_EXPLANATION),
    (EXAMPLE_3, EXAMPLE_3_COT_ACTIVATION_RESPONSE, EXAMPLE_3_EXPLANATION),
]

def system(use_cot: bool = False, insert_as_prompt: bool = False):
    if insert_as_prompt:
        return [
            {
                "role": "user",
                "content": SYSTEM.format(prompt=""),
            },
            {
                "role": "assistant",
                "content": "Understood! Please provide examples for analysis.",
            }
        ]
    
    else:
        prompt = "\n"

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
        response = cot_response + "\n" + explanation if use_cot else explanation

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
    examples,
    use_cot: bool = False,
    insert_as_prompt: bool = False,
):
    messages = system(
        use_cot=use_cot,
        insert_as_prompt=insert_as_prompt,
    )

    few_shot_examples = build_examples(
        use_cot=use_cot,
    )

    messages.extend(few_shot_examples)

    user_start = f"\n{examples}\n"

    messages.append(
        {
            "role": "user",
            "content": user_start,
        }
    )

    return messages
