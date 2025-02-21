import asyncio
import os
import json
from collections import defaultdict

# from seed import set_seed
from transformers import AutoTokenizer
from neurondb import load_torch
from neurondb.autointerp import Explainer, OpenRouterClient
from tqdm.asyncio import tqdm

# set_seed(42)

EXPLAINER_MODEL = "meta-llama/llama-3.3-70b-instruct"

FEATURE_SAVE_DIR = "/share/u/caden/neurondb/experiments/crosscoders/outputs"
GENERATION_KWARGS = {
    "temperature": 0.5,
    "max_tokens": 2000,
}
PROVIDER_KWARGS = {
    "provider": {
        "order": [
            "Novita"  # bf16, not fp8 or fp16
        ]
    }
}

DATA_DIR = "/share/u/caden/neurondb/experiments/crosscoders"
FILE_NAMES = [
    "data_0_1025.pt",
    "data_1026_2050.pt",
    "data_2051_3067.pt",
    "data_3068_4183.pt",
    "data_4184_5294.pt",
    "data_5296_6400.pt",
    "data_6401_7404.pt",
    "data_7405_7788.pt",
]


async def main():
    subject_tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")

    client = OpenRouterClient(model=EXPLAINER_MODEL, max_retries=1)
    semaphore = asyncio.Semaphore(25)

    explainer = Explainer(
        client=client,
        subject_tokenizer=subject_tokenizer,
        threshold=0.3,
        insert_as_prompt=True,
        verbose=False,
    )

    async def process_feature(file_name, feature, explanations):
        async with semaphore:  # Implement semaphore here
            index = str(feature.index)
            explanation = await explainer(
                feature, extra_body=PROVIDER_KWARGS, **GENERATION_KWARGS
            )
            explanations[index] = explanation
            pbar.update(1)

    for file_name in FILE_NAMES:
        file_name = os.path.basename(file_name)
        feature_save_path = os.path.join(DATA_DIR, file_name)
        output_file = os.path.join(FEATURE_SAVE_DIR, f"explanations_{file_name}.json")
        
        features = [
            f
            for f in load_torch(
                feature_save_path,
                max_examples=2_000,
                ctx_len=128,
            )
        ]
        
        # Create/load existing explanations
        explanations = {}
        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                explanations = json.load(f)

        pbar = tqdm(desc=f"Explaining {file_name}", total=len(features))

        # Process features in batches of 100
        for i in range(0, len(features), 100):
            batch = features[i:i + 100]
            await asyncio.gather(
                *(process_feature(file_name, feature, explanations) for feature in batch)
            )
            
            # Save after each batch of 100
            with open(output_file, "w") as f:
                json.dump(explanations, f)

            break

        pbar.close()


if __name__ == "__main__":
    asyncio.run(main())
