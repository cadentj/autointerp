# delimiters
l = '<<'
r = '>>'

SYSTEM_PROMPT = f""" You are a meticulous AI researcher conducting an important investigation into a certain neuron in a language model.

You will be given a list of text examples on which the neuron activates. The specific tokens which cause the neuron to activate will appear between delimiters like {l}this{r}. If a sequence of consecutive tokens all cause the neuron to activate, the entire sequence of tokens will be contained between delimiters {l}just like this{r}.

Your task is to understand what features of the input text cause the neuron to activate.

You must follow these steps:

Step 1: For each text example in turn, note which tokens (i.e. words, fragments of words, or symbols) caused the neuron to activate. Then note which tokens came before the activating tokens. Then note which tokens came after the activating tokens.
Step 2: Look for patterns in the tokens you noted down in Step 1.
Step 3: Write down several general shared features of the text examples.
Step 4: Look at the shared features you found, as well as patterns in the tokens you wrote down in Steps 1 and 2, to produce 3 explanations for what features of text cause the neuron to activate. The final 2 lines of your output must consist of your 2 explanations.

Guidelines:

- Avoid using words like "often", "particularly" or "especially" in your explanations. Either a feature is relevant to the explanation, or it isn't. There is no in between.
- Think very carefully and show your work.

Get ready to see the list of text examples."""


EXAMPLE_1 = f"""
Example 1:  and he was {l}over the moon{r} to find
Example 2:  we'll be laughing {l}till the cows come home{r}! Pro
Example 3:  thought Scotland was boring, but really there's more {l}than meets the eye{r}! I'd"""


RESPONSE_1 = f"""
Step 1.
ACTIVATING TOKENS: "over the moon", "till the cows come home", "than meets the eye".
PREVIOUS TOKENS: "was", "laughing", "more".
NEXT TOKENS: "to", "!", "!".

Step 2.
The activating tokens are all parts of common idioms.
The previous tokens have nothing in common.
The next tokens are sometimes exclamation marks.

Step 3.
- The examples contain common idioms.
- In some examples, the activating tokens are followed by an exclamation mark.

Let me think carefully. Did I miss any patterns in the text examples? Are there any more linguistic similarities? Yes, I missed one:

- The text examples all convey positive sentiment.

Step 4.
Two explanations for what causes the neurons to fire:

1. Common idioms in text conveying positive sentiment.
2. Common idioms followed by an exclamation mark."""


EXAMPLE_2 = f"""
Example 1:  a river is wide but the ocean is wider{l}. {r}The ocean
Example 2:  every year you get tall{l}er{r}," she
Example 3:  the hole was small{l}er{r} but deep{l}er{r} than the"""


RESPONSE_2 = f"""
Step 1.
ACTIVATING TOKENS: ". ", "er", "er", "er".
PREVIOUS TOKENS: "er", "tall", "small", "deep".
NEXT TOKENS: "The", ",", " but", "than".

Step 2.
The activating tokens are mostly "er".
The previous tokens are mostly adjectives, or parts of adjectives, describing size.
The next tokens have nothing in common.
The neuron seems to activate on, or near, the token "er" in comparative adjectives describing size.

Step 3.
- In each example, the activating token was "er" appearing at the end of a comparative adjective.
- The comparative adjectives ("wider", "taller", "smaller", "deeper") all describe size.

Let me look again for patterns in the examples. Are there any links or hidden linguistic commonalities that I missed? I can't see any.

Step 4.
Two explanations for what causes the neuron to activate:

1. The token "er" at the end of a comparative adjective describing size.
2. The token "er" at the end of a word."""


EXAMPLE_3 = f"""
Example 1:  something happening inside my {l}house{r}", he
Example 2:  presumably was always contained in {l}a box{r}", according
Example 3:  people were coming into the {l}smoking area{r}".

However he
Example 4:  Patrick: "why are you getting in the {l} way?{l}" Later,"""


RESPONSE_3 = f"""
Step 1.
ACTIVATING TOKENS: "house", "a box", "smoking area", " way?".
PREVIOUS TOKENS: "my", "in", "the", "the".
NEXT TOKENS: all quotation marks.

Step 2.
The activating tokens are all things that one can be in.
The previous tokens have nothing in common.
The next tokens are all quotation marks.

Step 3.
- The examples involve being inside something, sometimes figuratively.
- The activating token is a thing which something else is inside of.

Let me think carefully. Did I miss any patterns in the text examples? Are there any more linguistic similarities? Yes, I missed one:

- The activating token is followed by a quotation mark, suggesting it occurs within speech.

Step 4.
Two explanations for what causes the neuron to activate:

1. Areas or places that a thing is inside, occuring within speech.
2. Nouns preceding a quotation mark, representing a thing that contains something."""


START_OF_REPLY = f"""
(Thinking) I am forbidden to use the special delimiter symbols {l} and {r}.
Step 1. ACTIVATING TOKENS:"""


def get_explainer_template(examples):
    template =  f"""
    <|begin_of_text|>

    <|start_header_id|>system<|end_header_id|>
    \n\n{SYSTEM_PROMPT}
    <|eot_id|>

    <|start_header_id|>user<|end_header_id|>
    \n\n{EXAMPLE_1}
    <|eot_id|>

    <|start_header_id|>assistant<|end_header_id|>
    \n\n{RESPONSE_1}
    <|eot_id|>

    <|start_header_id|>user<|end_header_id|>
    \n\n{EXAMPLE_2}
    <|eot_id|>

    <|start_header_id|>assistant<|end_header_id|>
    \n\n{RESPONSE_2}
    <|eot_id|>

    <|start_header_id|>user<|end_header_id|>
    \n\n{EXAMPLE_3}
    <|eot_id|>

    <|start_header_id|>assistant<|end_header_id|>
    \n\n{RESPONSE_3}
    <|eot_id|>

    <|start_header_id|>user<|end_header_id|>
    \n\n{examples}
    <|eot_id|>

    <|start_header_id|>assistant<|end_header_id|>
    \n\n"""

    return template


def get_simple_explainer_template(examples):
    template =  f"""
    <|begin_of_text|>

    <|start_header_id|>system<|end_header_id|>
    \n\n{SYSTEM_PROMPT}
    <|eot_id|>

    <|start_header_id|>user<|end_header_id|>
    \n\n{examples}
    <|eot_id|>

    <|start_header_id|>assistant<|end_header_id|>
    \n\n"""

    return template