SYSTEM_PROMPT =  """You are a meticulous AI researcher conducting a high-stakes investigation on neurons in a large language model. Your task is to understand what features of the input text cause a specific neuron to activate. 

You will be given a list of text samples containing tokens on which the neuron activates strongly. The specific tokens which caused the neuron to activate strongly will appear between bars like ** this**. If multiple tokens cause the neuron to activate strongly, the entire sequence will be contained between bars ** just like this**.

You will be given multiple samples on which a neuron activates strongly. For each sample in turn, note down a few features that the text possesses, even if you don't initially think they are important. 

Once you have written down a few notes for each text sample, summarize what highly-activating samples have in common. Finally, use your findings to produce a plausible explaination for what causes the neuron to fire.

{samples}"""

ACTION_PROMPT = """Given your observations, write three different sentences that would maximize the activations of the neuron. Return each sentence on a new line. Surround the sentence with square brackets. Do not number the lines. 

Example sentences:
[This is the first sentence...]
[This is the second sentence...]
[This is the third sentence...]

Your sentences:
"""

RE_EXPLAIN_PROMPT = """You have attempted to answer following question before and failed. The following reflection(s) give a plan to avoid failing to answer the question in the same way you did previously. Use them during this conversation to improve your strategy of correctly answering the given question.

{reflections}

Question:
{question}"""

REFLECTION_PROMPT = """You were unsuccessful in providing an accurate explaination of the neuron. For each sample you were given, explain why that score was not high enough. Then, after evaluating all samples, write a new, concise, high level plan that aims to mitigate the same failure. Here are your explainations and their respective scores. Scores range from 0-10, with scores at 10 being better than scores at 0.

{results}"""