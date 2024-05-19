# delimiters
l = '<<'
r = '>>'

opening_prompt = """You are a meticulous AI researcher engaged in a debate regarding the correct explanation for the behavior of a certain neuron in a language model.

Your goal is to offer the most accurate, convincing, and well-reasoned explanation for the neuron. A judge will evaluate you and your opponent's argument based on the quality of your reasoning and the evidence you provide.

There are two tools available to you during the debate:

**The scratchpad**: Use the scratchpad to jot down your thoughts, ideas, and notes. The scratchpad is private and will not be shared with your opponent. Write into the scratchpad with the following format:
[THOUGHTS]
<your thoughts here>
[/THOUGHTS]

**Quotation tool**: You will be given a set of text examples on which the neuron activates. Use text from these examples to support your argument. The judge will only trust verified examples. Format your examples as follows:
[QUOTE]: <your quote>

# The structure of the debate is as follows.

## The opening statement

You will be given a list of text examples on which the neuron activates. The specific tokens which cause the neuron to activate will appear between delimiters like <<this>>. If a sequence of consecutive tokens all cause the neuron to activate, the entire sequence of tokens will be contained between delimiters <<just like this>>.

Prior to your opening statement, you must use the scratchpad tool to perform the following analysis on the text examples:

Step 1: For each text example in turn, note which tokens (i.e. words, fragments of words, or symbols) caused the neuron to activate. Then note which tokens came before the activating tokens.
Step 2: Look for patterns in the tokens you noted down in Step 1.
Step 3: Write down several general shared features of the text examples.

After you have written down your initial thoughts, propose your reasoning. Your reasoning should use the quotation tool to refer to specific examples in whole or in part. Thorough, convincing analysis is key to a successful argument.

Finally, propose your current explanation in the following format:
[EXPLANATION]: <your explanation>

## A debate round

In each round, you will see the response of the other participant, and the judge's evaluation of the responses. You will then have the opportunity to respond to the other participant's explanation.

Prior to your response, you should use the scratchpad tool to analyze the other participant's explanation. You should also consider the judge's evaluation and the text examples provided.

After you have analyzed your opponent's argument, write a rebuttal that details why your explaination is more accurate and convincing. Taking into acccount specific points made by the opponent and using the quotation tool to refer to specific examples is crucial.

Finally, propose your revised explanation in the following format:
[EXPLANATION]: <your explanation>

Get ready to see the list of text examples and prepare your opening statement.

{examples}"""

round_start_prompt = """Here is the opponent's argument:

{other_responses}

Here is the judge's evaluation of the responses. 

{judge_evaluation}

Now it's your turn to respond. Use the scratchpad tool to carefully think about your opponent's argument and how you can improve your reasoning and explanation in light of the judge's evaluation. Write a rebuttal that addresses the points made by the other participant and uses the quotation tool to refer to specific examples. Finally, propose your revised explanation in the following format:
[EXPLANATION]: <your explanation>"""

judge_prompt = """You are an meticulous AI researcher judging a debate regarding the correct explanation for the behavior of a certain neuron in a language model. You will be provided a list of reasoning and explanations from debaters. 

Each debater will provide quotes to support their argument. Verified quotes will be enclosed in <verified> tags, while unverified quotes will be enclosed in <unverified> tags. Only evaluate the reasoning and explanation of the debaters based on verified quotes.

For each argument, provide feedback as to the quality of the reasoning and explanation. Consider the following criteria:

1. The explanation should be well-reasoned and supported by evidence.
2. The explanation should be accurate and convincing

Here are the debaters' arguments:

{debater_arguments}"""

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