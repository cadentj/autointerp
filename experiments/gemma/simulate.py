import torch as t
import json
from collections import defaultdict

from transformers import AutoTokenizer
from neurondb import load_torch
from neurondb.autointerp import NsClient, simulate

t.set_grad_enabled(False)

EXPLAINER_MODEL = "meta-llama/llama-3.3-70b-instruct"
SIMULATOR_MODEL = "Qwen/Qwen2.5-14B-Instruct"

def main(save_dir, layers):
    subject_tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")

    client = NsClient(
        SIMULATOR_MODEL,
        # torch_dtype=t.bfloat16,
    )

    explanations_save_path = f"{save_dir}/explanations.json"
    with open(explanations_save_path, "r") as f:
        explanations = json.load(f)

    results = defaultdict(dict)

    for layer, layer_explanations in explanations.items():
        if int(layer) not in layers:
            continue

        feature_save_path = f"{save_dir}/.model.layers.{layer}.pt"
        for feature in load_torch(feature_save_path, max_examples=2_000, train=False, ctx_len=32):
            examples = feature.examples

            feature_index = feature.index
            explanation = layer_explanations.get(str(feature_index), False)

            if not explanation:
                print(f"No explanation found for feature {feature_index}")
                continue

            results[layer][str(feature_index)] = simulate(
                explanation,
                examples,
                client,
                subject_tokenizer,
                return_predictions=True,
            )

        with open(f"{save_dir}/l{layer}-scores.json", "w") as f:
            json.dump(results, f)

if __name__ == "__main__":
    save_dir = "/share/u/caden/neurondb/cache/gemma-2-2b"
    layers = list(range(13, 26))
    print(layers)
    main(save_dir, layers)