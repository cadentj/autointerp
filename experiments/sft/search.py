# %%

import json
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

scores = "/share/u/caden/neurondb/cache/steering_finetuning/llama-3.3-70b-instruct-simulated-by-Qwen2.5-7B-Instruct.json"


with open(scores, "r") as f:
    scores = json.load(f)

per_layer_scores = defaultdict(list)

for layer, layer_scores in scores.items():
    for feature_index, (per_example_score, total_score) in layer_scores.items():
        per_layer_scores[layer].append(total_score)

# %%

# Number of buckets
n_buckets = 4
layers = sorted([int(layer.split(".")[-1]) for layer in per_layer_scores.keys()])
bucket_size = len(layers) / n_buckets

# Create bucketed scores
bucketed_scores = defaultdict(list)
for layer, scores in per_layer_scores.items():
    layer_num = int(layer.split(".")[-1])
    bucket = int(layer_num // bucket_size)  # Integer division to get bucket index
    bucket_name = f'Layers {int(bucket * bucket_size)}-{int((bucket + 1) * bucket_size - 1)}'
    bucketed_scores[bucket_name].extend(scores)

plt.figure(figsize=(12, 6))

# Calculate KDE for each bucket
for bucket_name, scores in bucketed_scores.items():
    scores = np.array(scores)
    kernel = stats.gaussian_kde(scores)
    x_range = np.linspace(min(scores), max(scores), 200)
    plt.plot(x_range, kernel(x_range), label=bucket_name)

plt.title('Distribution of Scores by Layer Groups')
plt.xlabel('Score')
plt.ylabel('Density')
plt.legend()
plt.tight_layout()
plt.show()


# %%

with open("per_task_indices.json", "r") as f:
    per_task_indices = json.load(f)

explanations_path = "/share/u/caden/neurondb/cache/steering_finetuning/llama-3.3-70b-instruct-explanations.json"

with open(explanations_path, "r") as f:
    all_explanations = json.load(f)

def get_explanations(dataset_a, dataset_b):
    indices = per_task_indices[f"{dataset_a}_{dataset_b}"]
    
    explanations = {}
    for layer, indices in indices.items():
        for index in indices:
            explanations[(layer, index)] = all_explanations[layer][str(index)]

    return explanations

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("NovaSearch/stella_en_1.5B_v5", trust_remote_code=True).cuda()

def search(query, dataset_a, dataset_b):
    explanations = get_explanations(dataset_a, dataset_b)

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

sports_query = "Sports-related terms, activities, and concepts including team sports, athletic competitions, and professional leagues. This encompasses baseball concepts like pitching, batting, fielding, base running, innings, and statistics, along with famous baseball players throughout history. Basketball terminology including shooting, dribbling, defense, offensive plays, court positions, and legendary basketball players and teams. Football concepts covering offensive and defensive strategies, player positions, scoring methods, game rules, historic NFL moments, and notable football athletes. Sports equipment, venues, championship events, training methods, coaching philosophies, and memorable sports moments."

pronouns_query = "Words and phrases related to personal pronouns and their various forms in language usage. This includes subject pronouns like he, she, they, it, we, and you; object pronouns such as him, her, them, us, and me; possessive pronouns including his, hers, theirs, ours, and yours; reflexive pronouns like himself, herself, themselves, itself, and ourselves. Also covers demonstrative pronouns like this, that, these, and those; relative pronouns such as who, whom, whose, which, and that; and indefinite pronouns including anyone, everyone, someone, no one, and anybody. Includes gender-specific, gender-neutral, and collective pronoun usage in various contexts."

verbs_query = "Grammatical elements focusing on the relationship between nouns and verbs in sentence construction. This encompasses singular and plural noun forms, regular and irregular verb conjugations, and proper verb usage patterns. Includes various forms of common verbs like have/has, am/is/are, go/goes, do/does, and their past tense variations. Covers special cases like collective nouns, compound subjects, and intervening phrases between subjects and verbs. Also includes grammatical number agreement, tense consistency, and proper usage of helping verbs. Features sentence structures demonstrating both standard and exceptional cases of matching singular and plural subjects with their corresponding verb forms, including complex noun phrases and coordinated subjects."

sentiment_query = "Words, phrases, and expressions that convey emotional tone, attitude, and feeling in language. This includes strongly positive terms expressing joy, excitement, approval, and satisfaction; negative terms conveying disappointment, anger, sadness, and disapproval; and neutral language for objective or balanced expression. Encompasses emotional intensifiers, modifiers, and qualifiers that strengthen or soften sentiment. Includes context-dependent emotional markers, idiomatic expressions with positive or negative connotations, and words that describe various degrees of emotional response. Covers both explicit sentiment indicators and subtle emotional undertones in language usage."

sports = "hc-mats/sports-gemma-2-2b-top-1000"
pronouns = "kh4dien/mc-gender"
verbs = "hc-mats/subject-verb-agreement"
sentiment = "kh4dien/mc-sentiment"

# See paper figure
ablations = {
    (verbs, sentiment) : verbs_query,
    (sports, pronouns) : pronouns_query,
    (pronouns, sports) : sports_query,
    (sentiment, verbs) : sentiment_query,
    (sentiment, sports) : sports_query,
    (verbs, sports) : sports_query,
    (sentiment, pronouns) : pronouns_query,
    (verbs, pronouns) : pronouns_query,
}

results = {}

for (a, b), query in ablations.items():
    similarities = search(query, a, b)
    sorted_similarities = sorted(similarities.items(), key=lambda x: x[1]["similarity"], reverse=True)
    filtered_similarities = {k:v for k,v in sorted_similarities if v["similarity"] > 0.5}
    results[(a, b)] = filtered_similarities


# %%

for (a, b), similarities in results.items():
    a = a.split("/")[-1]
    b = b.split("/")[-1]
    with open(f"{a}_{b}.json", "w") as f:    
        features = defaultdict(list)
        for key in similarities:
            features[key[0]].append(key[1])
        json.dump(features, f)


# %%

# similarities = search(query, a, b)
import matplotlib.pyplot as plt

# horizontal bar of number of explanations per ablations
labels = []
values = []
for (a, b), similarities in results.items():
    labels.append(f"{a}_{b}")
    values.append(len(similarities))

plt.figure(figsize=(10, 6))
plt.barh(labels, values)
plt.xlabel("Number of Explanations")
plt.tight_layout()
plt.show()
