# %%
from autointerp.vis.dashboard import make_dashboard
from sparsify import Sae

path = "/workspace/qwen-saes/layers.31"
sae = Sae.load_from_disk(path, device="cuda")

cache_path = "/workspace/qwen-cache/model.layers.31"
dashboard = make_dashboard(cache_path, sae.simple_encode)

# hey how are you doing?

# %%

