import asyncio
from autointerp.automation import OpenRouterClient, Classifier
from autointerp import load, make_quantile_sampler

from tqdm import tqdm
import json
import os
import torch as t

SCORER_MODEL = "meta-llama/Llama-3.3-70B-Instruct"
# NOTE: Should add 1 back in later
FEATURE_PATHS = [
    f"/share/u/caden/autointerp/experiments/other/group_{i}.pt" for i in [0, 1, 2, 3]
]
BATCH_SIZE = 100
SCORES_PATH = "/share/u/caden/autointerp/experiments/other_outputs/llama_scores.json"
EXPLANATIONS_PATH = (
    "/share/u/caden/autointerp/experiments/other_outputs/llama_explanations.json"
)
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


async def score():
    client = OpenRouterClient(SCORER_MODEL)
    scorer = Classifier(client=client, method="detection", n_examples_shown=5)
    os.makedirs(os.path.dirname(SCORES_PATH), exist_ok=True)

    semaphore = asyncio.Semaphore(20)

    sampler = make_quantile_sampler(n_examples=30, n_quantiles=3, n_exclude=10)

    explanations = json.load(open(EXPLANATIONS_PATH, "r"))

    for path in FEATURE_PATHS:
        features = load(path, sampler, load_non_activating=20)
        pbar = tqdm(total=len(features))

        for batch in range(0, len(features), BATCH_SIZE):
            feature_batch = features[batch : batch + BATCH_SIZE]

            async def process(feature):
                async with semaphore:
                    return await scorer(
                        feature, explanations[str(feature.index)], **KWARGS
                    )

            tasks = [
                process(feature) for feature in feature_batch
            ]

            batch_scores = await asyncio.gather(*tasks)
            batch_scores = {
                feature.index: score
                for feature, score in zip(feature_batch, batch_scores)
            }
            write_scores(batch_scores)

            pbar.update(len(feature_batch))

        #     break
        # break

        pbar.close()


if __name__ == "__main__":
    asyncio.run(score())
