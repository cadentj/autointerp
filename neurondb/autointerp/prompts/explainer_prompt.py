### SYSTEM PROMPT ###

SYSTEM = """You are a meticulous AI researcher conducting an important investigation into patterns found in language. Your task is to analyze text and provide an explanation that thoroughly encapsulates possible patterns found in it.

You will be given a list of text examples on which special words are selected and between delimiters like <<this>>. If a sequence of consecutive tokens all are important, the entire sequence of tokens will be contained between delimiters <<just like this>>. How important each token is for the behavior is listed after each example in parentheses. 

Rules:
- Produce a concise, one or two sentence final description. Simply describe the text features that are common in the examples, and what patterns you found.
- If the examples are uninformative, you don't need to mention them. Don't focus on giving examples of important tokens, but try to summarize the patterns found in the examples.
- Do not mention the marker tokens (<< >>) in your explanation.
- Assume someone reading your explanation does not have access to the examples you saw. You should not point to specific examples in your explanation.
- The last line of your response must be the formatted explanation, using \\boxed{Your explanation here.}."""

### EXAMPLE 1 ###

EXAMPLE_1 = """Example 1:  and he was <<over the moon>> to find
Example 2:  we'll be laughing <<till the cows come home>>! Pro
Example 3:  thought Scotland was boring, but really there's more <<than meets the eye>>! I'd"""

EXAMPLE_1_EXPLANATION = """\\boxed{Common idioms in text conveying positive sentiment.}"""

### EXAMPLE 2 ###

EXAMPLE_2 = """Example 1:  a river is wide but the ocean is wid<<er>>. The ocean
Example 2:  every year you get tall<<er>>," she
Example 3:  the hole was small<<er>> but deep<<er>> than the"""

EXAMPLE_2_EXPLANATION = """\\boxed{The token "er" at the end of a comparative adjective describing size.}"""

### EXAMPLE 3 ###

EXAMPLE_3 = """Example 1:  something happening inside my <<house>>", he
Example 2:  presumably was always contained in <<a box>>", according
Example 3:  people were coming into the <<smoking area>>".

However he
Example 4:  Patrick: "why are you getting in the << way?>>" Later,"""

EXAMPLE_3_EXPLANATION = """\\boxed{Nouns representing distinct objects that contain something, sometimes preceding a quotation mark.}"""

EXAMPLES = [
    (EXAMPLE_1, EXAMPLE_1_EXPLANATION),
    (EXAMPLE_2, EXAMPLE_2_EXPLANATION),
    (EXAMPLE_3, EXAMPLE_3_EXPLANATION),
]

def system(insert_as_prompt: bool = False):
    if insert_as_prompt:
        return [
            {
                "role": "user",
                "content": SYSTEM,
            },
            {
                "role": "assistant",
                "content": "Understood! Please provide examples for analysis.",
            }
        ]
    
    else:

        return [
            {
                "role": "system",
                "content": SYSTEM,
            }
        ]

def build_examples():
    examples = []

    for example, explanation in EXAMPLES:
        messages = [
            {
                "role": "user",
                "content": example,
            },
            {
                "role": "assistant",
                "content": explanation,
            },
        ]

        examples.extend(messages)

    return examples


def build_prompt(
    examples,
    insert_as_prompt: bool = False,
):
    messages = system(
        insert_as_prompt=insert_as_prompt,
    )

    few_shot_examples = build_examples()

    messages.extend(few_shot_examples)

    messages.append(
        {
            "role": "user",
            "content": examples,
        }
    )

    return messages
