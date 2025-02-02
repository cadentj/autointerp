from neurondb import load_torch
from neurondb.autointerp import Explainer, LocalClient
from transformers import AutoTokenizer
import asyncio
import json
import os


async def main():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")

    client = LocalClient(model="Qwen/Qwen2.5-1.5B-Instruct", max_retries=2)

    explainer = Explainer(client, tokenizer=tokenizer)

    # Create dict to store results
    explanations = {}

    async def process_feature(feature):
        explanation = await explainer(feature)
        explanations[feature.index] = explanation
        print(f"Processed feature {feature.index}")

    # Create and gather all tasks
    tasks = [
        process_feature(feature)
        for feature in load_torch(
            "/share/u/caden/neurondb/cache/.model.layers.0.pt", 
            max_examples=50
        )
    ]
    await asyncio.gather(*tasks)

    # Create output directory if it doesn't exist
    os.makedirs('outputs', exist_ok=True)
    
    # Save the explanations to a JSON file
    output_path = 'outputs/explanations.json'
    with open(output_path, 'w') as f:
        json.dump(explanations, f, indent=2)
    
    print(f"Saved {len(explanations)} explanations to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
