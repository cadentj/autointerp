from prompt_builder import Activation

### EXPLAINER PROMPTS ###

system_prompt = """We're studying neurons in a neural network. Each neuron looks for some particular thing in a short document. Look at the parts of the document the neuron activates for and summarize in a single sentence what the neuron is looking for. Don't list examples of words.

The activation format is token<tab>activation. Activation values range from 0 to 10. A neuron finding what it's looking for is represented by a non-zero activation value. The higher the activation value, the stronger the match."""

few_shot_explainations = [
    {
        "acts" : Activation(
        "explaination" : "The neuron is looking for a person."
    },
]