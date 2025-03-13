# %%

import json

with open("./llama_scores.json", "r") as f:
    batch_topk_scores = json.load(f)

with open("../crosscoder/llama_scores.json", "r") as f:
    crosscoder_scores = json.load(f)

import torch as t
top_indices = t.load("/share/u/caden/autointerp/experiments/batch_topk/chat_only_indices.pt").tolist()

batch_topk_scores = {k: v for k, v in batch_topk_scores.items() if int(k) in top_indices}


# %%

import numpy as np
import matplotlib.pyplot as plt

def compute_scores(scores):
    random_scores = [s["-1"]["TN"] for s in scores.values()]
    similar_scores = [s["0"]["TN"] for s in scores.values()]
    value_scores = [[s[str(i)]["TP"] for s in scores.values()] for i in range(1,4)]

    # Mean, stf
    line = [
        (np.mean(random_scores) / 20, np.std(random_scores) / 20),
        (np.mean(similar_scores) / 20, np.std(similar_scores) / 20),
        (np.mean(value_scores[0]) / 10, np.std(value_scores[0]) / 10),
        (np.mean(value_scores[1]) / 10, np.std(value_scores[1]) / 10),
        (np.mean(value_scores[2]) / 10, np.std(value_scores[2]) / 10),
    ]

    return line


def plot_grouped(batch_scores, crosscoder_scores):
    batch_results = compute_scores(batch_scores)
    crosscoder_results = compute_scores(crosscoder_scores)

    labels = ["Random", "Similar", "Value-1", "Value-2", "Value-3"]
    x_pos = np.arange(len(labels))
    width = 0.35  # Width of the bars

    batch_means = [x[0] for x in batch_results]
    batch_stds = [x[1] for x in batch_results]
    crosscoder_means = [x[0] for x in crosscoder_results]
    crosscoder_stds = [x[1] for x in crosscoder_results]

    plt.figure(figsize=(12, 6))
    plt.bar(x_pos - width/2, batch_means, width, yerr=batch_stds, 
            label='Batch TopK', alpha=0.8, ecolor='black', capsize=5)
    plt.bar(x_pos + width/2, crosscoder_means, width, yerr=crosscoder_stds,
            label='CrossCoder', alpha=0.8, ecolor='black', capsize=5)

    plt.xlabel('Method')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Comparison: Batch TopK vs CrossCoder')
    plt.xticks(x_pos, labels)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# Replace the individual plots with the grouped plot
plot_grouped(batch_topk_scores, crosscoder_scores)


# %%

