import asyncio
from autointerp.automation import OpenRouterClient, Explainer
from autointerp import load

EXPLAINER_MODEL = "meta-llama/Llama-3.3-70B-Instruct"
FEATURE_PATH = "/root/autointerp/cache/model.layers.0.pt"

async def explain():
    client = OpenRouterClient(EXPLAINER_MODEL)
    explainer = Explainer(client=client)

    features = load(FEATURE_PATH)
    tasks = [
        explainer(feature)
        for feature in features
    ]
    explanations = await asyncio.gather(*tasks)
    print(explanations)

if __name__ == "__main__":
    asyncio.run(explain())

