# %%
import json
import os
from neurondb.autointerp.tune import (
    SimulatorDataset,
    prepare_tokenizer,
)
from transformers import AutoTokenizer

import torch as t
from datasets import Dataset, DatasetDict

def set_seed(seed: int):
    import random
    import numpy as np
    
    random.seed(seed)
    t.manual_seed(seed)
    t.cuda.manual_seed(seed)
    t.cuda.manual_seed_all(seed)
    t.backends.cudnn.deterministic = True
    t.backends.cudnn.benchmark = False
    np.random.seed(seed)

GEMMA_CACHE = "/share/u/caden/neurondb/cache/gemma-2-2b"
SIMULATOR_MODEL = "Qwen/Qwen2.5-3B"
set_seed(42)

def load_raw_data():
    with open(os.path.join(GEMMA_CACHE, "explanations.json"), "r") as f:
        explanations = json.load(f)

    scores = {}
    for file in os.listdir(GEMMA_CACHE):
        if "-scores.json" in file:
            with open(os.path.join(GEMMA_CACHE, file), "r") as f:
                raw_scores = json.load(f)

            for layer, layer_scores in raw_scores.items():
                scores[layer] = layer_scores

    path_template = (
        "/share/u/caden/neurondb/cache/gemma-2-2b/.model.layers.{layer}.pt"
    )
    activation_paths = {
        layer: path_template.format(layer=layer) for layer in scores.keys()
    }

    return explanations, scores, activation_paths

def load(simulator_tokenizer):
    explanations, scores, activation_paths = load_raw_data()

    subject_tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
    simulator_tokenizer = prepare_tokenizer(simulator_tokenizer)

    dataset = SimulatorDataset(
        explanations=explanations,
        scores=scores,
        subject_tokenizer=subject_tokenizer,
        simulator_tokenizer=simulator_tokenizer,
        activation_paths=activation_paths,
    )

    return dataset, simulator_tokenizer

tokenizer = AutoTokenizer.from_pretrained(SIMULATOR_MODEL)
dataset, simulator_tokenizer = load(tokenizer)

# %%

from collections import defaultdict

data = defaultdict(lambda: defaultdict(list))

for example in dataset:
    layer = example['layer']
    feature = example['feature_index']
    
    data[layer][feature].append(example)

train = defaultdict(list)
test = defaultdict(list)

for layer, features in data.items():
    for feature in features.keys():
        examples = features[feature]
        if train.get(layer, None) is None:
            train[layer].append(examples)
        else:
            n_train = len(train[layer])
            if n_train >= 80:
                test[layer].append(examples)
            else:
                train[layer].append(examples)

# %%

train_flattened = sum(train.values(), [])
train_flattened = sum(train_flattened, [])
test_flattened = sum(test.values(), [])
test_flattened = sum(test_flattened, [])


len(train_flattened)

train_dataset = Dataset.from_list(train_flattened)
test_dataset = Dataset.from_list(test_flattened)

dataset = DatasetDict({
    "train": train_dataset,
    "test": test_dataset,
})

dataset.push_to_hub("kh4dien/explainer-gemma-2_simulator-qwen2.5")