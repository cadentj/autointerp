# %%
from typing import List, Dict, Any
from autointerp.vis.dashboard import make_dashboard


dashboard = make_dashboard(
    "/root/autointerp/cache/model.layers.0",
    lambda x: x,
)

dashboard.display()
# %%


