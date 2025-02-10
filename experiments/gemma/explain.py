import asyncio
import os
import json

from collections import defaultdict
from seed import set_seed
from transformers import AutoTokenizer
from neurondb import load_torch
from neurondb.autointerp import Explainer, OpenRouterClient
from tqdm.asyncio import tqdm

set_seed(42)


EXPLAINER_MODEL = "meta-llama/llama-3.3-70b-instruct"
SUBJECT_MODEL = "google/gemma-2-2b"
FEATURE_SAVE_DIR = "/share/u/caden/neurondb/cache/gemma-2-2b"
GENERATION_KWARGS = {
    "temperature": 0.5,
    "max_tokens": 2000,
}
PROVIDER_KWARGS = {
    "provider" : {
        "order" : [
            "Novita" # bf16, not fp8 or fp16
        ]
    }
}

async def main():
    subject_tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")

    client = OpenRouterClient(model=EXPLAINER_MODEL, max_retries=1)
    semaphore = asyncio.Semaphore(25)

    explainer = Explainer(
        client=client,
        subject_tokenizer=subject_tokenizer,
        threshold=0.5,
        insert_as_prompt=True,
        verbose=False
    )

    # Create save path for explanations
    explanations_path = "explanations.json"
    save_path = os.path.join(FEATURE_SAVE_DIR, explanations_path)

    # Load existing explanations if file exists
    explanations = defaultdict(dict)
    if os.path.exists(save_path):
        with open(save_path, "r") as f:
            explanations.update(json.load(f))

    async def process_feature(layer, feature):
        index = str(feature.index)

        if layer in explanations and index in explanations[layer]:
            pbar.update(1)
            return
        
        async with semaphore:
            explanation = await explainer(feature, extra_body=PROVIDER_KWARGS, **GENERATION_KWARGS)
            explanations[layer][index] = explanation
            
            # Save every 10 features
            if int(index) % 10 == 0:
                with open(save_path, "w") as f:
                    json.dump(explanations, f)
            
            pbar.update(1)

    for file in os.listdir(FEATURE_SAVE_DIR):
        if not file.endswith(".pt"):
            continue

        layer = file.replace(".pt", "").replace(".model.layers.", "")

        feature_save_path = os.path.join(FEATURE_SAVE_DIR, file)
        features = [f for f in load_torch(feature_save_path, max_examples=2_000, ctx_len=32)]
        pbar = tqdm(desc=f"Layer {layer}", total=len(features))

        # Process features directly from generator
        await asyncio.gather(
            *(
                process_feature(layer, feature)
                for feature in features
            )
        )

    pbar.close()

if __name__ == "__main__":
    asyncio.run(main())
