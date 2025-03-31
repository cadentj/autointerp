# %%
from autointerp.vis.dashboard import make_dashboard
from sparsify import Sae

path = (
    "/workspace/gemma-saes/gemma-3-4b-step-final/language_model.model.layers.16"
)
sae = Sae.load_from_disk(path, device="cuda")


cache_path = "/workspace/gemma-cache/language_model.model.layers.16"
dashboard = make_dashboard(cache_path, sae.simple_encode, in_memory=False)

# hey how are you doing?

# %%


from nnsight import LanguageModel
import torch as t

model = LanguageModel(
    "google/gemma-3-4b-pt",
    device_map="auto",
    dispatch=True,
    torch_dtype=t.bfloat16,
)

with model.trace(
    "hey how are you doing?",
) as ret:
    acts = model.language_model.model.layers[16].output[0].save()


latents = sae.simple_encode(acts)

# %%
