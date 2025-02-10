# %%

# Dataset definitions
sports = "hc-mats/sports-gemma-2-2b-top-1000"
pronouns = "kh4dien/mc-gender"
verbs = "hc-mats/subject-verb-agreement"
sentiment = "kh4dien/mc-sentiment"

# Ablation pairs and whether to use flipped accuracy
ablations = {
    (verbs, sentiment): 1,
    (sports, pronouns): 0,
    (pronouns, sports): 0,
    (sentiment, verbs): 1,
    (sentiment, sports): 0,
    (verbs, sports): 0,
    (sentiment, pronouns): 0,
    (verbs, pronouns): 0,
}

name_map = {
    sports : "sports",
    pronouns : "pronouns",
    verbs : "sva",
    sentiment : "sentiment",
}

# %%


import os
import json

def get_name(a, b, which):
    a_name = a.split("/")[-1]
    b_name = b.split("/")[-1]
    caden_name = f"{a_name}_{b_name}.json"

    choice = a if which == 1 else b

    part_one = name_map[choice]

    a = name_map[a]
    b = name_map[b]

    helena_name = f"mcmc_{part_one}_interpreted_top_100_{a}_{b}_intervention_kwargs.json"

    return caden_name, helena_name

caden_dir = "/share/u/caden/neurondb/experiments/sft/ablations"
helena_dir = "/share/u/caden/neurondb/experiments/sft/helena"

for (a, b), which in ablations.items():
    caden_name, helena_name = get_name(a, b, which)

    caden_path = os.path.join(caden_dir, caden_name)
    helena_path = os.path.join(helena_dir, helena_name)

    caden_kwargs = json.load(open(caden_path))
    helena_kwargs = json.load(open(helena_path))['feats_to_ablate']

    for layer, helena_feats in helena_kwargs.items():
        caden_layer = f".{layer}"

        caden_feats = caden_kwargs.get(caden_layer, [])

        diff = set(caden_feats) - set(helena_feats)

        extra_helena = len(diff)

        if extra_helena > 0:
            print("YOU DIFFERED AT")
            print("CADEN: ", caden_name)
            print("HELENA: ", helena_name)
            print("LAYER: ", layer)
            print("HEL: ", helena_feats)
            print("CAD: ", caden_feats)
            print("DIFF: ", diff)
            break


    