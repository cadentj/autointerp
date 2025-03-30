from autointerp.vis.dashboard import make_dashboard
from sparsify import Sae

path = "/workspace/gemma-3-4b-saes/gemma-3-4b-step-final/language_model.model.layers.16"
sae = Sae.load_from_disk(path, device="cuda")

cache_path = "/root/autointerp/cache/language_model.model.layers.16"
dashboard = make_dashboard(cache_path, sae.simple_encode)

dashboard.display()
