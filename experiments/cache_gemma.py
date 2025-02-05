import os 
import argparse
import torch as t

from models.gemma import load_gemma
from neurondb import cache_activations

from seed import set_seed

set_seed(42)
FEATURE_IDXS = list(range(250))

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-size", type=str)
    parser.add_argument("--width", type=str)
    parser.add_argument("--l0", type=int)
    parser.add_argument("--layer", type=int)
    return parser.parse_args()

def main(args):
    model, submodules = load_gemma(
        model_size=args.model_size,
        width=args.width,
        l0=args.l0,
        layers = [args.layer],
        torch_dtype=t.bfloat16,
    )

    token_save_dir = "/workspace"
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

    save_dir = f"/workspace/cache/gemma-2-{args.model_size}-w{args.width}-l0{args.l0}-layer{args.layer}"
    os.makedirs(save_dir, exist_ok=True)

    cache.save_to_disk(
        save_dir,
        token_save_path
    )

if __name__ == "__main__":
    args = get_args()
    main(args)