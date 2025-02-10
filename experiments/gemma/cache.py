import os 
import torch as t

import sys

sys.path.append("/share/u/caden/neurondb/experiments")

from models.gemma import load_gemma
from neurondb import cache_activations

from seed import set_seed

set_seed(42)
FEATURE_IDXS = list(range(100))

def main():
    model, submodules = load_gemma()

    token_save_dir = "/share/u/caden/neurondb/cache"
    token_save_path = os.path.join(token_save_dir, "tokens.pt")
    tokens = t.load(token_save_path)

    cache = cache_activations(
        model,
        {sm.module : sm.dictionary for sm in submodules},
        tokens,
        max_tokens=5_000_000,
        batch_size=16,
        filters={sm.module._path : FEATURE_IDXS for sm in submodules}
    )

    save_dir = "/share/u/caden/neurondb/cache/gemma-2-2b"
    os.makedirs(save_dir, exist_ok=True)

    cache.save_to_disk(
        save_dir,
        token_save_path
    )

if __name__ == "__main__":
    main()