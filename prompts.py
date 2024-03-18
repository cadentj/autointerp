SYSTEM_PROMPT =  """We're studying neurons in a neural network. Each neuron looks for some particular thing in a short document. Look at the parts of the document the neuron activates for and think step by step about what the neuron is doing. 

The activation format is token<tab>activation. Activation values range from 0 to 10. A neuron finding what it's looking for is represented by a non-zero activation value. The higher the activation value, the stronger the match.

(EXAMPLE USER)
Sentence: abcdef

Activations:
<start>
a	10
b	0
c	0
d	0
e	10
f	0
<end>

(EXAMPLE MODEL)
- Activations are high on letters a and e which are vowels
- Letters for which activations are low are consonants
- This neuron activates on vowels

(USER)
Sentence: {sentence}

Activations:
<start>
{activations}
<end>

(MODEL)
"""

ACTION_PROMPT = """Now, write three sentences that would maximize the activations of the neuron. Return your sentences in brackets.

(EXAMPLE MODEL)
{sentence1}
{sentence2}
{sentence3}

(MODEL)
"""

RE_EXPLAIN_PROMPT = "Given the self reflections..."

REFLECTION_PROMPT = """You will be given the history of a past experience in which you were placed in an environment and given a task to complete. You were unsuccessful in completing the task. Do not summarize your environment, but rather think about the strategy and path you took to attempt to complete the task. Devise a concise, new plan of action that accounts for your mistake with reference to specific actions that you should have taken. For example, if you tried A and B but forgot C, then devise a plan to achieve C with environment-specific actions. You will need this later when you are solving the same task. Give your plan after "Plan". Here are two examples:

{trial}

Given your score, """

