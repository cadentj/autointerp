# %%

from autointerp.vis.dashboard import make_dashboard
from sparsify import Sae

path = (
    "/workspace/gemma-saes/gemma-3-4b-step-final/language_model.model.layers.16"
)
sae = Sae.load_from_disk(path, device="cuda")


cache_path = "/workspace/gemma-cache/language_model.model.layers.16"
dashboard = make_dashboard(cache_path, sae.simple_encode, in_memory=False)

# %%

from autointerp.vis.dashboard import make_dashboard
from sparsify import Sae

path = "/workspace/qwen-saes-two/qwen-step-final/model.layers.47"
sae = Sae.load_from_disk(path, device="cuda")

cache_path = "/workspace/qwen-cache/model.layers.47"
dashboard = make_dashboard(cache_path, sae.simple_encode, in_memory=True)

# %%

from autointerp.vis.dashboard import make_feature_display

cache_path = "/workspace/qwen-ssae-cache-two/model.layers.31"

features = list(range(100))
feature_display = make_feature_display(cache_path, features)
