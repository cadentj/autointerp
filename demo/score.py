import asyncio
from autointerp.automation import OpenRouterClient, Classifier
from autointerp import load, make_quantile_sampler

SCORER_MODEL = "meta-llama/Llama-3.3-70B-Instruct"
FEATURE_PATH = "/root/autointerp/cache/model.layers.0.pt"

async def score():
    client = OpenRouterClient(SCORER_MODEL)
    scorer = Classifier(client=client, method="detection")
    
    sampler = make_quantile_sampler(
        n_examples=20, n_quantiles=10, n_top_exclude=20
    )
    features = load(FEATURE_PATH, sampler)
    tasks = [scorer(feature, "some explanation here") for feature in features]
    scores = await asyncio.gather(*tasks)
    print(scores)

if __name__ == "__main__":
    asyncio.run(score())
