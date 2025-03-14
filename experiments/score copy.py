# %%

from autointerp.automation import Classifier
from autointerp import load, make_quantile_sampler

import json
import os

import random

WHICH = "batch_topk"
FEATURE_PATHS = [
    f"/share/u/caden/autointerp/experiments/{WHICH}/group_{i}.pt"
    for i in [0, 1, 2, 3]
]
EXPLANATIONS_PATH = f"/share/u/caden/autointerp/experiments/outputs/{WHICH}/llama_explanations.json"

ROOT = "/share/u/caden/autointerp/experiments/"

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.3-70B-Instruct")

def score():
    scorer = Classifier(client=None, method="detection", n_examples_shown=5)
    sampler = make_quantile_sampler(n_examples=10, n_quantiles=3, n_exclude=10)

    explanations = json.load(open(EXPLANATIONS_PATH, "r"))

    features = load(
        FEATURE_PATHS[0],
        sampler,
        load_similar_non_activating=20,
        load_random_non_activating=20,
    )

    random.seed(42)
    test_features = random.sample(features, 10)

    for feature in test_features:
        index = feature.index
        explanation = explanations[str(index)]
        prompts, truth, quantiles = scorer(feature, explanation)

        os.makedirs(f"{ROOT}manual/feature_{index}", exist_ok=True)

        for i, (prompt, t, q) in enumerate(zip(prompts, truth, quantiles)):
            print(q)

            with open(f"{ROOT}manual/feature_{index}/prompt_{i}.json", "w") as f:
                json.dump({"prompt": prompt, "truth": t, "quantiles": q}, f)


# %%

score()