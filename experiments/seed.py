from datasets import load_dataset

def set_seed(seed: int):
    import random
    import numpy as np
    import torch as t
    
    random.seed(seed)
    t.manual_seed(seed)
    t.cuda.manual_seed(seed)
    t.cuda.manual_seed_all(seed)
    t.backends.cudnn.deterministic = True
    t.backends.cudnn.benchmark = False
    np.random.seed(seed)

print("SETTING SEED")
set_seed(42)

def get_tokens(tokenizer):
    data = load_dataset("kh4dien/fineweb-100m-sample", split="train[:25%]")

    tokens = tokenizer(
        data["text"],
        padding=True,
        return_tensors="pt",
        truncation=True,
        max_length=1025, # 1024 + 1 for BOS
    )
    tokens = tokens["input_ids"]

    og_shape = tokens.shape[0]
    mask = ~(tokens == 0).any(dim=1)
    tokens = tokens[mask]
    print(f"Removed {og_shape - tokens.shape[0]} rows containing pad tokens")

    return tokens