# %%
from typing import List, Dict, Any
from autointerp.vis.dashboard import make_dashboard


dashboard = make_dashboard(
    "/root/autointerp/cache/model.layers.0",
    lambda x: x,
)

dashboard.display()
# %%


from autointerp.autointerp.vis.backend import Backend

path = "/root/autointerp/cache/google/gemma-3-4b-pt"
backend = Backend(cache_dir="/root/autointerp/cache/google/gemma-3-4b-pt")