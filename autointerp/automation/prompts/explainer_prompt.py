### SYSTEM PROMPT ###

SYSTEM = """You are a meticulous AI researcher conducting an important investigation into patterns found in language. Your task is to analyze text and provide an explanation that thoroughly encapsulates possible patterns found in it.

You will be given a list of text examples on which special words are selected and between delimiters like <<this>>. If a sequence of consecutive tokens all are important, the entire sequence of tokens will be contained between delimiters <<just like this>>. How important each token is for the behavior is listed after each example in parentheses. 

Rules:
- Produce a concise, one or two sentence final description. Simply describe the text features that are common in the examples, and what patterns you found.
- If the examples are uninformative, you don't need to mention them. Don't focus on giving examples of important tokens, but try to summarize the patterns found in the examples.
- Do not mention the marker tokens (<< >>) in your explanation, they're just for highlighting.
- Assume someone reading your explanation does not have access to the examples you saw. You should not point to specific examples in your explanation.
- Explanations will be sorted by the strength of their activations. Activation strength is sometimes helpful for understanding a feature; for example, a feature represents a model's uncertainty may activate more strongly when the model is more unsure.
- The last line of your response must be the formatted explanation, using \\boxed{Your explanation here.}."""

### EXAMPLE 1 ###



EXAMPLE_1 = """High Activating: 
Example 1: and he was <<over the moon>> to find 
Example 2: we'll be laughing <<till the cows come home>>! Pro 
Example 3: thought Scotland was boring, but really there's more <<than meets the eye>>! I'd

Middle Activating: 
Example 1: she felt like she had <<hit the jackpot>> when she got accepted 
Example 2: we had fun <<until the sun went down>>! 
Example 3: the event had much more <<than you'd expect at first glance>>!

Low Activating: 
Example 1: and he was <<really happy>> to find 
Example 2: we'll be laughing <<for a long time>>! 
Example 3: thought Scotland was boring, but really there's <<a lot to see>>!"""

EXAMPLE_1_EXPLANATION = """\\boxed{Common idioms in text conveying positive sentiment.}"""

### EXAMPLE 2 ###

# EXAMPLE_2 = """Example 1: a river is wide but the ocean is wid<<er>>. The ocean
# Example 2: every year you get tall<<er>>," she
# Example 3: the hole was small<<er>> but deep<<er>> than the"""

# EXAMPLE_2_EXPLANATION = """\\boxed{The token "er" at the end of a comparative adjective describing size.}"""

### EXAMPLE 3 ###

EXAMPLE_3 = """High Activating:
Example 1: presumably was always contained in <<a box>>", according
Example 2: people were coming into the <<smoking area>>".
Example 3: Patrick: "why are you getting in the <<way?>>" Later,

Middle Activating:
Example 1: something happening inside my <<tent>>", he
Example 2: presumably was always contained in <<a crate>>", according
Example 3: people were coming into the <<waiting room>>".

Low Activating:
Example 1: something happening inside my <<garden>>", he
Example 2: presumably was always contained in <<a pile>>", according
Example 3: people were coming into the <<corner>>"."""

EXAMPLE_3_EXPLANATION = """\\boxed{Nouns representing distinct objects that contain something, sometimes preceding a quotation mark.}"""

EXAMPLES = [
    (EXAMPLE_1, EXAMPLE_1_EXPLANATION),
    # (EXAMPLE_2, EXAMPLE_2_EXPLANATION),
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


TEMPLATE = """High Activating:
{high_activating_examples}

Middle Activating:
{middle_activating_examples}

Low Activating:
{low_activating_examples}"""

def build_prompt(
    high_activating_examples,
    middle_activating_examples,
    low_activating_examples,
    insert_as_prompt: bool = False,
):
    messages = system(
        insert_as_prompt=insert_as_prompt,
    )

    few_shot_examples = build_examples()

    messages.extend(few_shot_examples)

    examples = TEMPLATE.format(
        high_activating_examples=high_activating_examples,
        middle_activating_examples=middle_activating_examples,
        low_activating_examples=low_activating_examples,
    )

    messages.append(
        {
            "role": "user",
            "content": examples,
        }
    )

    return messages
