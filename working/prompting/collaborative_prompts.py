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
Step 4: Look at the shared features you found, as well as patterns in the tokens you wrote down in Steps 1 and 2, to produce a single explanation for what features of text cause the neuron to activate. Propose your explanation in the following format: 
[EXPLANATION]: <your explanation>

Guidelines:

- Avoid using words like "often", "particularly" or "especially" in your explanations. Either a feature is relevant to the explanation, or it isn't. There is no in between.
- Think very carefully and show your work.
- Please don't use extra formatting like markdown.

Here are several example explanations."""


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
[EXPLANATION]: Common idioms in text conveying positive sentiment."""


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
[EXPLANATION]: The token "er" at the end of a comparative adjective describing size."""


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
[EXPLANATION]: Nouns preceding a quotation mark, representing a thing that contains something."""

AGENT_START = f"""Now, it's your turn to propose an argument. Here is a list of text examples:

{{examples}}"""

opening_prompt = f"{SYSTEM_PROMPT}\n{EXAMPLE_1}\n{RESPONSE_1}\n{EXAMPLE_2}\n{RESPONSE_2}\n{EXAMPLE_3}\n{RESPONSE_3}\n{AGENT_START}"

round_start_prompt = """Here are the processes from other agents:
[OTHER RESPONSES]
{other_responses}
[/OTHER RESPONSES]

Now it's your turn to respond. Closely examine your reasoning and the reasoning of other agents and provide an updated explanation for the neuron's activation. Propose your explanation in the following format: 
[EXPLANATION]: <your explanation>"""

examples = """Example 1: reconstruction��, but because I��<<m>> evidence-based and
Example 2: and innovate on until we create something that we all appreciate. I��<<m>> excited to see
what
Example 3: of the May 1968 graffitists wrote: ��I��<<m>> not a servant of
Example 4: the city,�� Rep. Jackson said. ��I��<<m>> sure the mayor is
Example 5: bad year in Augusta, I said to myself, ��I��<<m>> sick of this.
Example 6: Nothing against those that hold belts, and fight big shows. I��<<m>> sure those achievements
are
Example 7: , right? How are you approaching combat in this one? I��<<m>> guessing you��
Example 8: ocks. It was a pretty high-achieving school. I��<<m>> not"""



