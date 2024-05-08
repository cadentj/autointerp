import sys
import os
import threading
from typing import List

import torch as t
from nnsight import LanguageModel

from environment import Environment
from keys import REPLICATE

sys.path.append("../mats_sae_training")
from sae_training.sparse_autoencoder import SparseAutoencoder

t.set_grad_enabled(False)

def run(
    provider: str, # openai or replicate
    layer: int,
    features: List[int],
):
    if provider == "replicate":
        os.environ["REPLICATE_API_TOKEN"] = REPLICATE

    os.environ["PROVIDER"] = provider

    # Load model and sae
    model = LanguageModel("openai-community/gpt2", device_map='auto', dispatch=True)
    sae = load_sae(layer)

    # Load configs for respective providers
    if provider == "openai":
        from config import EnvConfig, ExplainerConfig, CondenserConfig, DetectionScorerConfig, GenerationScorerConfig
    else:
        from llama_config import EnvConfig, ExplainerConfig, CondenserConfig, DetectionScorerConfig, GenerationScorerConfig

    # Load default configs for now
    env_config = EnvConfig()
    explainer_cfg = ExplainerConfig()
    condenser_cfg = CondenserConfig()
    d_scorer_cfg = DetectionScorerConfig()
    gen_scorer_cfg = GenerationScorerConfig()

    # Define a function to run each feature in a separate thread
    # def feature_thread(environment, feature_id, explainer_cfg, condenser_cfg, d_scorer_cfg, gen_scorer_cfg):
    #     environment.execute(feature_id, explainer_cfg, condenser_cfg, d_scorer_cfg, gen_scorer_cfg)
  
    # env = Environment(model, sae=sae, cfg=env_config, provider="openai")
    # env.load_cache(layer)

    # threads = []
    # for feature_id in features: 
    #     thread = threading.Thread(
    #         target=feature_thread, 
    #         args=(env, feature_id, explainer_cfg, condenser_cfg, d_scorer_cfg, gen_scorer_cfg)
    #     )
    #     threads.append(thread)
    #     thread.start()

    # for thread in threads:
    #     thread.join()
  
    env = Environment(model, sae=sae, cfg=env_config, provider=provider)
    env.load_cache(layer)

    for feature_id in features: 
        state = env.load_id(feature_id, explainer_cfg, condenser_cfg, d_scorer_cfg, gen_scorer_cfg)
        env.run_feature(state)

    del env, model, sae


def load_sae(
    layer: int
):
    local_dir = "../jbloom_dictionaries"
    filename =  f"final_sparse_autoencoder_gpt2-small_blocks.{layer}.hook_resid_pre_24576.pt"

    save_path = f"{local_dir}/{filename}"
    sae = SparseAutoencoder.load_from_pretrained(save_path)
    sae.to("cuda:0")

    return sae

# run("openai", 1, [1, 6243, 19991, 8434, 8231])

run("openai", 10, [11, 19, 33, 20, 7000])

run("replicate", 9, [23735, 23839, 24007, 24386, 21503])

run("replicate", 8, [16955, 15269, 10277, 14407, 6236])

