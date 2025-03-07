import asyncio
from autointerp.automation import OpenRouterClient, Explainer
from autointerp import load, make_quantile_sampler

from tqdm import tqdm
import json
import os

EXPLAINER_MODEL = "meta-llama/Llama-3.3-70B-Instruct"
FEATURE_PATHS = [
    f"/share/u/caden/autointerp/experiments/group_{i}.pt" for i in [0, 1, 2, 3]
]
BATCH_SIZE = 100
EXPLANATIONS_PATH = "/share/u/caden/autointerp/experiments/outputs/explanations.json"
KWARGS = {
    "provider" : {
        "order" : ["DeepInfra"]
    },
    "temperature" : 0.6,
    "max_tokens" : 500
}


def write_explanations(explanations):
    # Read existing explanations if file exists
    existing_explanations = {}
    if os.path.exists(EXPLANATIONS_PATH):
        with open(EXPLANATIONS_PATH, 'r') as f:
            existing_explanations = json.load(f)
    
    # Update with new explanations
    existing_explanations.update(explanations)
    
    # Write all explanations back to file
    with open(EXPLANATIONS_PATH, 'w') as f:
        json.dump(existing_explanations, f)

async def explain():
    client = OpenRouterClient(EXPLAINER_MODEL)
    explainer = Explainer(client=client)
    os.makedirs(os.path.dirname(EXPLANATIONS_PATH), exist_ok=True)

    semaphore = asyncio.Semaphore(20)

    sampler = make_quantile_sampler(n_examples=20, n_quantiles=1)

    for path in FEATURE_PATHS:
        features = load(path, sampler)
        pbar = tqdm(total=len(features))

        for batch in range(0, len(features), BATCH_SIZE):
            feature_batch = features[batch : batch + BATCH_SIZE]

            async def process_with_semaphore(feature):
                async with semaphore:
                    return await explainer(feature, **KWARGS)

            tasks = [process_with_semaphore(feature) for feature in feature_batch]
            
            batch_explanations = await asyncio.gather(*tasks)
            batch_explanations = {
                feature.index: explanation
                for feature, explanation in zip(
                    feature_batch, batch_explanations
                )
            }
            write_explanations(batch_explanations)

            pbar.update(len(feature_batch))

        pbar.close()

if __name__ == "__main__":
    asyncio.run(explain())
