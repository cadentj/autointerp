import os
import json
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import asyncio

from neurondb import load_torch, Feature
from neurondb.autointerp import Classifier, OpenRouterClient
from transformers import AutoTokenizer
from typing import Dict, List

# Constants
CLASSIFIER_MODEL = "meta-llama/llama-3.3-70b-instruct"
DATA_DIR = "/share/u/caden/neurondb/experiments/crosscoders"
FEATURE_SAVE_DIR = "/share/u/caden/neurondb/experiments/crosscoders/outputs"
SCORES_DIR = "/share/u/caden/neurondb/experiments/crosscoders/scores"

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

GENERATION_KWARGS = {
    "temperature": 0.0,
    "max_tokens": 100,
}


async def score_feature(
    classifier: Classifier,
    feature: Feature,
    explanation: str
) -> Dict:
    """Score a single feature's explanation"""
    outputs = await classifier(feature, explanation)
    return calculate_metrics(outputs)

async def score_file(
    file_name: str,
    classifier: Classifier,
    scores_dir: str,
    feature_dir: str,
    explanations_dir: str,
) -> None:
    """Score all features in a file"""
    base_name = os.path.basename(file_name)
    feature_path = os.path.join(feature_dir, base_name)
    explanation_path = os.path.join(explanations_dir, f"explanations_{base_name}.json")
    scores_path = os.path.join(scores_dir, f"scores_{base_name}.json")
    
    if os.path.exists(scores_path):
        print(f"Skipping {base_name} - already processed")
        return
        
    with open(explanation_path, 'r') as f:
        explanations = json.load(f)
        
    features = list(load_torch(
        feature_path,
        max_examples=2_000,
        ctx_len=128,
        train=False,
        indices=[0]
    ))
    
    scores = {}
    for feature in tqdm(features, desc=f"Scoring {base_name}"):
        feature_id = str(feature.index)
        if feature_id not in explanations:
            continue
            
        explanation = explanations[feature_id]
        scores[feature_id] = await score_feature(classifier, feature, explanation)
        
        # Break after first feature
        print(f"\nScored feature {feature_id}:")
        print(scores[feature_id])
        break
        
    with open(scores_path, 'w') as f:
        json.dump(scores, f, indent=2)
        
    # Calculate and print average scores
    avg_scores = defaultdict(float)
    for feature_scores in scores.values():
        for metric, value in feature_scores.items():
            avg_scores[metric] += value
    
    for metric in avg_scores:
        avg_scores[metric] /= len(scores)
        
    print(f"\nAverage scores for {base_name}:")
    for metric, value in avg_scores.items():
        print(f"{metric}: {value:.3f}")

async def main():
    os.makedirs(SCORES_DIR, exist_ok=True)
    
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
    client = OpenRouterClient(model=CLASSIFIER_MODEL)
    classifier = Classifier(
        client=client,
        tokenizer=tokenizer,
        n_examples_shown=5,
        method="detection",
        log_prob=False,
        temperature=0.0,
    )

    for file_name in FILE_NAMES:
        await score_file(
            file_name,
            classifier,
            SCORES_DIR,
            DATA_DIR,
            FEATURE_SAVE_DIR
        )

if __name__ == "__main__":
    asyncio.run(main()) 

