import torch as t
import argparse
import json

from transformers import AutoTokenizer
from neurondb import load_torch
from neurondb.autointerp import NsClient, simulate

t.set_grad_enabled(False)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-size", type=str)
    parser.add_argument("--width", type=str)
    parser.add_argument("--l0", type=int)
    parser.add_argument("--layer", type=int)
    parser.add_argument("--explainer-model", type=str)
    parser.add_argument("--simulator-model", type=str)
    return parser.parse_args()

def save_result(subject_tokenizer, examples, per_example_expected_values, explainer_model_name, save_dir):
    def _format_results(tokens, example_evs, true_acts):
        lines = []
        for token, ev, act in zip(tokens, example_evs, true_acts):
            lines.append(f"{token}\t\t{ev:.2f}\t{act}")
        return "\n".join(lines)

    tokens = [subject_tokenizer.batch_decode(example.tokens) for example in examples]
    result = [
        _format_results(toks, evs, example.normalized_activations.tolist())
        for toks, evs, example in zip(tokens, per_example_expected_values, examples)
    ]

    with open(f"{save_dir}/{explainer_model_name}-scores.txt", "w") as f:
        f.write("\n\n\n".join(result))


def main(args, save_dir):
    subject_tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")

    client = NsClient(
        args.simulator_model,
        torch_dtype=t.bfloat16,
    )

    explainer_model_name = args.explainer_model.split("/")[-1]
    explanations_save_path = f"{save_dir}/{explainer_model_name}-explanations.json"

    with open(explanations_save_path, "r") as f:
        explanations = json.load(f)[str(args.layer)]

    results = {}

    feature_save_path = f"{save_dir}/.model.layers.{args.layer}.pt"
    for feature in load_torch(feature_save_path, max_examples=2_000):
        examples = feature.examples

        feature_index = feature.index
        explanation = explanations.get(str(feature_index), False)

        if not explanation:
            print(f"No explanation found for feature {feature_index}")
            continue

        results[str(feature_index)] = simulate(
            explanation,
            examples,
            client,
            subject_tokenizer,
        )

    simulator_model_name = args.simulator_model.split("/")[-1]
    with open(f"{save_dir}/{explainer_model_name}-simulated-by-{simulator_model_name}.json", "w") as f:
        json.dump(results, f)

if __name__ == "__main__":
    args = get_args()

    save_dir = f"/root/neurondb/cache/gemma-2-{args.model_size}-w{args.width}-l0{args.l0}-layer{args.layer}"

    main(args, save_dir)
