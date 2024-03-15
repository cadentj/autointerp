### EXPLAINER PROMPTS ###

base_prompt = """We're studying neurons in a neural network. Each neuron looks for some particular thing in a short document. Look at the parts of the document the neuron activates for and summarize in a single sentence what the neuron is looking for. Don't list examples of words.

The activation format is token<tab>activation. Activation values range from 0 to 10. A neuron finding what it's looking for is represented by a non-zero activation value. The higher the activation value, the stronger the match.

Neuron 1
Activations:
<start>
a	10
b	0
c	0
<end>
<start>
d	0
e	10
f	0
<end>

Explanation of neuron 1 behavior: the main thing this neuron does is find vowels.

Neuron 2
Activations:
<start>
a	10
b	0
c	0
<end>
<start>
d	0
e	10
f	0
<end>

Explanation of neuron 2 behavior:<|endofprompt|> the main thing this neuron does is find"""

few_shot_one = """one
"""

few_shot_two = """two
"""

few_shot_three = """three
"""

few_shot_four = """four
"""

### SIMULATOR PROMPTS ###

