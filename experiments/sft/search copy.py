# %%

import json
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

explanations_path = "/share/u/caden/neurondb/cache/gender/llama-3.3-70b-instruct-explanations.json"
gender_indices = "/share/u/caden/neurondb/experiments/sft/gender_indices.json"

with open(explanations_path, "r") as f:
    all_explanations = json.load(f)

def get_explanations():
    with open(gender_indices, "r") as f:
        indices = json.load(f)

    explanations = {}
    for layer, indices in indices.items():
        for index in indices:
            layer_explanations = all_explanations.get(layer, {})
            if layer_explanations:
                explanations[(layer, index)] = layer_explanations.get(str(index), "")

    return explanations

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("NovaSearch/stella_en_1.5B_v5", trust_remote_code=True).cuda()

def search(query):
    explanations = get_explanations()

    # This model supports two prompts: "s2p_query" and "s2s_query" for sentence-to-passage and sentence-to-sentence tasks, respectively.
    # They are defined in `config_sentence_transformers.json`
    query_prompt_name = "s2p_query"

    query_embeddings = model.encode([query], prompt_name=query_prompt_name)
    doc_embeddings = model.encode(list(explanations.values()))

    similarities = model.similarity(query_embeddings, doc_embeddings)[0]

    return {
        key : {
            "explanation" : explanations[key],
            "similarity" : similarities[i].item()
        }
        for i, key in enumerate(explanations.keys())
    }

# %%
query = "Gender or profession related terms, especially gendered towards women, females, doctors, nurses, etc."
similarities = search(query)
sorted_similarities = sorted(similarities.items(), key=lambda x: x[1]["similarity"], reverse=True)
filtered_similarities = {k:v for k,v in sorted_similarities if v["similarity"] > 0.5}
results = filtered_similarities

# %%

per_layer = defaultdict(list)

for key, value in results.items():
    per_layer[key[0]].append(key[1])


with open("gender_results.json", "w") as f:
    json.dump(per_layer, f)

# %%

