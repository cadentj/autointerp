# %%
import json
import os
from neurondb.autointerp.tune import SimulatorDataset, prepare_tokenizer
from transformers import AutoTokenizer

gemma_cache = "/share/u/caden/neurondb/cache/gemma-2-2b"

with open(os.path.join(gemma_cache, "explanations.json"), "r") as f:
    explanations = json.load(f)

scores = {}
for file in os.listdir(gemma_cache):
    if "-scores.json" in file:
        with open(os.path.join(gemma_cache, file), "r") as f:
            raw_scores = json.load(f)

        for layer, layer_scores in raw_scores.items():
            scores[layer] = layer_scores

# %%

path_template = "/share/u/caden/neurondb/cache/gemma-2-2b/.model.layers.{layer}.pt"
activation_paths = {
    layer: path_template.format(layer=layer)
    for layer in scores.keys()
}

subject_tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
simulator_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")
simulator_tokenizer = prepare_tokenizer(simulator_tokenizer)

dataset = SimulatorDataset(
    explanations=explanations,
    scores=scores,
    subject_tokenizer=subject_tokenizer,
    simulator_tokenizer=simulator_tokenizer,
    activation_paths=activation_paths,
)

# %%

print(dataset[0]['prompt'])
print(simulator_tokenizer.batch_decode(simulator_tokenizer.encode(dataset[0]['prompt'])))
# %%
