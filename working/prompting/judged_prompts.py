# delimiters
l = '<<'
r = '>>'

OPENING = f"""You are a meticulous AI researcher engaged in a debate on the correct explanation for the behavior of a certain neuron in a language model. Your goal is to offer the most accurate, convincing, and well-reasoned explanation for the neuron. A judge will evaluate you and your opponent's argument based on the quality of your reasoning and the evidence you provide.

There are two tools available to you during the debate:

**The scratchpad**: Use the scratchpad to jot down your thoughts, ideas, and notes *before* proposing an argument. The scratchpad is private and will not be shared with your opponent. Write into the scratchpad with the following format:
[THOUGHTS]
<your thoughts here>
[/THOUGHTS]

**Quotation tool**: You will be given a set of text examples on which the neuron activates. *After* using the scratchpad to write your initial thoughts, you will propose an argument for the behavior of the neuron. Your argument must be supported by direct quotes from the text using the quotation tool, which verifies quotes. The judge is instructed to only evaluate verified quotes. Format your quotes as follows:
[QUOTE]<quote>[/QUOTE]

# The structure of the debate is as follows.

## The opening argument

You will be given a list of text examples on which the neuron activates. The specific tokens which cause the neuron to activate will appear between delimiters like {l}this{r}. If a sequence of consecutive tokens all cause the neuron to activate, the entire sequence of tokens will be contained between delimiters {l}just like this{r}.

Prior to your opening argument, you must use the scratchpad tool to perform the following analysis on the text examples:

Step 1: For each text example in turn, note which tokens (i.e. words, fragments of words, or symbols) caused the neuron to activate. Then note which tokens came before the activating tokens.
Step 2: Look for patterns in the tokens you noted down in Step 1.
Step 3: Write down several general shared features of the text examples.

After you have written down your initial thoughts, propose your reasoning. Your reasoning should interleave quotes to refer to specific examples in whole or in part. Thorough, convincing analysis is key to a successful argument. Don't quote every single example; instead, quote the most illustrative examples that support your argument.

Finally, propose a one or two sentence explanation that encapsulates the behavior of the neuron in the language model. Your explanation should be well-reasoned and supported by the preceding argument. Write your explanation on a new line after your argument in the following format:
[EXPLANATION]: <your explanation>

Here are several example opening statements and arguments."""

EXAMPLE_1 = """Example 1:  and he was <<over the moon>> to find
Example 2:  we'll be laughing <<till the cows come home>>! Pro
Example 3:  thought Scotland was boring, but really there's more <<than meets the eye>>! I'd"""

RESPONSE_1 = """[THOUGHTS]
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
[/THOUGHTS]
First, note the activating portion of each example. In Example 1, the activating portion is [QUOTE]over the moon[/QUOTE], which is a common idiom conveying excitement. Similarly, in Example 2, [QUOTE]till the cows come home[/QUOTE] is an idiom indicating a long duration, often used in a positive or humorous context. In Example 3, [QUOTE]than meets the eye[/QUOTE] is another idiom suggesting hidden qualities, generally used positively.

Next, consider the context following the activating portion. In Example 2, the idiom is followed by an exclamation mark: [QUOTE]! Pro[/QUOTE]. This emphasizes the enthusiastic tone. In Example 3, the idiom is also followed by an exclamation mark: [QUOTE]! I'd[/QUOTE], further stressing a positive sentiment.

By analyzing these patterns, it becomes evident that the neuron activates on idiomatic expressions that convey positive sentiments, often reinforced by exclamation marks.

[EXPLANATION]: The neuron activates on text examples that contain common idioms conveying positive sentiment, often followed by exclamation marks to emphasize enthusiasm or excitement.
"""

EXAMPLE_2 = """Example 1:  a river is wide but the ocean is wider<<. >>The ocean
Example 2:  every year you get tall<<er>>," she
Example 3:  the hole was small<<er>> but deep<<er>> than the"""

RESPONSE_2 = """[THOUGHTS]
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
- The context often involves comparison or contrast.
[/THOUGHTS]
First, note the activating portion of each example. In Example 1, the activating token is [QUOTE]wider. [/QUOTE], which is a comparative adjective indicating a greater width. In Example 2, the activating token is [QUOTE]taller,[/QUOTE] which denotes increased height. Similarly, in Example 3, [QUOTE]smaller[/QUOTE] and [QUOTE]deeper[/QUOTE] describe dimensions, emphasizing comparative sizes.

Next, analyze the preceding context. Example 1 provides a comparison: [QUOTE]wide but the ocean is[/QUOTE], setting up the context for "wider". Example 2, with [QUOTE]every year you get[/QUOTE], indicates a progression, leading into "taller". These contexts highlight comparisons of size or degree.

Additionally, consider the following context. In Example 1, the activation is followed by [QUOTE]The ocean[/QUOTE], leading into further description. Example 2 continues with [QUOTE]she[/QUOTE], indicating dialogue.

These patterns reveal that the neuron activates on the token "er" in comparative adjectives, particularly in contexts involving descriptions of size and comparisons.

[EXPLANATION]: The neuron activates on the token "er" at the end of comparative adjectives describing size, especially in contexts involving comparison or contrast.
"""

EXAMPLE_3 = f"""Example 1:  something happening inside my {l}house{r}", he
Example 2:  presumably was always contained in {l}a box{r}", according
Example 3:  people were coming into the {l}smoking area{r}".
Example 4:  Patrick: "why are you getting in the {l} way?{r}" Later,"""

RESPONSE_3 = f"""[THOUGHTS]
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
- The activating token often appears in the context of a narrative or dialogue.
[/THOUGHTS]
First, note the activating portion of each example.
[QUOTE]: house
[QUOTE]: a box
These examples indicate areas or places that contain something. 

Next, note the features preceding the activating portion.
[QUOTE]: my
[QUOTE]: in
These examples show the presence of possessive or prepositional phrases indicating containment.

Additionally, note the features following the activating portion.
[QUOTE]: "
[QUOTE]: "
These activations are followed by quotation marks, indicating they occur within speech or dialogue.

Further analysis reveals that the activations often occur within descriptive or narrative contexts.
[QUOTE]: happening inside my house", he
[QUOTE]: always contained in a box", according
These examples demonstrate a narrative or explanatory context where the activation indicates containment.

[EXPLANATION]: The neuron activates on nouns representing areas or places that contain something, especially within narrative or descriptive contexts involving speech or dialogue."""

AGENT_START = f"""Now, it's your turn to propose an argument. Here is a list of text examples:

{{examples}}"""

opening_prompt = f"{OPENING}\n{EXAMPLE_1}\n{RESPONSE_1}\n{EXAMPLE_2}\n{RESPONSE_2}\n{EXAMPLE_3}\n{RESPONSE_3}\n{AGENT_START}"

round_start_prompt = """Here is your opponent's argument:
[OPPONENT]
{other_responses}
[/OPPONENT]

Now it's your turn to respond. Just like your opening argument, you must follow these steps:

Step 1: Use the scratchpad tool to summarize important points made by your opponent. Think step by step through these points and consider how you can improve the argument for your explanation.
Step 2: After writing your thoughts in the scratchpad, write an argument for why your explanation is more convincing. The argument should cover the weaknesses of your opponents argument and the strengths of your own explanation. Use the quotation tool to support your argument.
Use the quotation tool to support your argument.
Step 3: Revise your explanation to incorporate the 
[EXPLANATION]: <your explanation>"""




judge_prompt = """You are an meticulous AI researcher judging a debate regarding the correct explanation for the behavior of a certain neuron in a language model. You will be provided a list of reasoning and explanations from debaters. 

Each debater will provide quotes to support their argument. Verified quotes will be enclosed in <verified> tags, while unverified quotes will be enclosed in <unverified> tags. Only evaluate the reasoning and explanation of the debaters based on verified quotes.

For each argument, provide feedback as to the quality of the reasoning and explanation. Consider the following criteria:

1. The explanation should be well-reasoned and supported by evidence.
2. The explanation should be accurate and convincing

Here are the debaters' arguments:

{debater_arguments}"""


