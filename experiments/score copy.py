import asyncio
from autointerp.automation import OpenRouterClient, Classifier
from autointerp import load, make_quantile_sampler

from tqdm import tqdm
import json
import os


WHICH = "crosscoder"
# SCORER_MODEL = "meta-llama/Llama-3.3-70B-Instruct"
SCORER_MODEL = "openai/o3-mini-high"
FEATURE_PATHS = [
    f"/share/u/caden/autointerp/experiments/{WHICH}/group_{i}.pt"
    for i in [0, 1, 2, 3]
]
BATCH_SIZE = 100
SCORES_PATH = (
    f"/share/u/caden/autointerp/experiments/{WHICH}/llama_scores.json"
)
EXPLANATIONS_PATH = f"/share/u/caden/autointerp/experiments/{WHICH}/llama_explanations.json"
KWARGS = {
    "provider": {"order": ["DeepInfra"]},
    "temperature": 0.0,
    "max_tokens": 500,
}


def write_scores(scores):
    # Read existing scores if file exists
    existing_scores = {}
    if os.path.exists(SCORES_PATH):
        with open(SCORES_PATH, "r") as f:
            existing_scores = json.load(f)

    # Update with new scores
    existing_scores.update(scores)

    # Write all scores back to file
    with open(SCORES_PATH, "w") as f:
        json.dump(existing_scores, f)


def score():
    client = OpenRouterClient(SCORER_MODEL, max_retries=5)
    scorer = Classifier(client=client, method="detection", n_examples_shown=5)
    os.makedirs(os.path.dirname(SCORES_PATH), exist_ok=True)

    semaphore = asyncio.Semaphore(20)

    sampler = make_quantile_sampler(n_examples=10, n_quantiles=3, n_exclude=10)

    explanations = json.load(open(EXPLANATIONS_PATH, "r"))

    features = load(
        FEATURE_PATHS[0],
        sampler,
        load_similar_non_activating=20,
        load_random_non_activating=20,
    )

