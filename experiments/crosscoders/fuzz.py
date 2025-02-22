import os
import json
from collections import defaultdict
from tqdm import tqdm
import asyncio

from neurondb import load_torch
from neurondb.autointerp import Classifier, OpenRouterClient
from transformers import AutoTokenizer

# Constants
CLASSIFIER_MODEL = "meta-llama/llama-3.3-70b-instruct"
FEATURES_DIR = "/share/u/caden/neurondb/experiments/crosscoders"
EXPLANATIONS_DIR = "/share/u/caden/neurondb/experiments/crosscoders/outputs"
SCORES_DIR = "/share/u/caden/neurondb/experiments/crosscoders/fuzzing"
SCORING_TYPE = "fuzzing"

FILE_NAMES = [
    "data_0_1025.pt",
    "data_1026_2050.pt",
    "data_2051_3067.pt",
    # "data_3068_4183.pt",
    # "data_4184_5294.pt",
    # "data_5296_6400.pt",
    # "data_6401_7404.pt",
    # "data_7405_7788.pt",
]

GENERATION_KWARGS = {
    "temperature": 0.0,
    "max_tokens": 100,
    "provider": {
        "order": [
            "Lambda"
        ]
    }
}

DETECTION_KWARGS = {
    "n_random": 20,
    "n_quantiles": 10,
    "n_test": 40,
    "train": False,
}

FUZZING_KWARGS = {
    "n_random": 0, 
    "n_quantiles": 10,
    "n_test": 60,
    "train": False,
}

async def score_feature(
    feature,
    feature_id: str,
    explanation: str,
    classifier: Classifier,
    semaphore: asyncio.Semaphore,
) -> tuple[str, dict]:
    """Score a single feature with semaphore control"""
    async with semaphore:
        score = await classifier(feature, explanation)
        return feature_id, score

async def score_file(
    file_name: str,
    classifier: Classifier,
    scoring_type: str,
    scores_dir: str,
    feature_dir: str,
    explanations_dir: str,
) -> None:
    """Score all features in a file"""
    base_name = os.path.basename(file_name)
    feature_path = os.path.join(feature_dir, base_name)
    base_name = base_name.replace(".pt", "")
    explanation_path = os.path.join(explanations_dir, f"explanations_{base_name}.json")
    scores_path = os.path.join(scores_dir, f"scores_{base_name}.json")
    
    # Load existing scores if they exist
    scores = {}
    if os.path.exists(scores_path):
        with open(scores_path, 'r') as f:
            scores = json.load(f)
        print(f"Loaded {len(scores)} existing scores from {base_name}")
        
    with open(explanation_path, 'r') as f:
        explanations = json.load(f)

    kwargs = DETECTION_KWARGS if scoring_type == "detection" else FUZZING_KWARGS
    features = list(load_torch(
        feature_path,
        max_examples=2_000,
        ctx_len=64,
        **kwargs
    ))
    
    # Create tasks for features that need scoring
    semaphore = asyncio.Semaphore(25)
    tasks = []
    for feature in tqdm(features, desc=f"Creating tasks for {base_name}"):
        feature_id = str(feature.index)
        if feature_id not in explanations or feature_id in scores:
            continue
            
        tasks.append(score_feature(
            feature,
            feature_id,
            explanations[feature_id],
            classifier,
            semaphore
        ))
    
    # Process features in batches of 100
    total_tasks = len(tasks)
    pbar = tqdm(total=total_tasks, desc=f"Processing features from {base_name}")
    for i in range(0, total_tasks, 100):
        batch = tasks[i:i + 100]
        results = await asyncio.gather(*batch)
        
        # Update scores with batch results
        for feature_id, score in results:
            scores[feature_id] = score
            
        # Save progress
        with open(scores_path, 'w') as f:
            json.dump(scores, f)
            
        pbar.update(len(batch))
    
    pbar.close()
    
    # Final save
    with open(scores_path, 'w') as f:
        json.dump(scores, f)

async def main():
    os.makedirs(SCORES_DIR, exist_ok=True)
    
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
    client = OpenRouterClient(model=CLASSIFIER_MODEL, max_retries=2)
    classifier = Classifier(
        client=client,
        tokenizer=tokenizer,
        n_examples_shown=5,
        method=SCORING_TYPE,
        verbose=False,
    )

    for file_name in FILE_NAMES:
        await score_file(
            file_name,
            classifier,
            SCORING_TYPE,
            SCORES_DIR,
            FEATURES_DIR,
            EXPLANATIONS_DIR,
        )

if __name__ == "__main__":
    asyncio.run(main()) 

