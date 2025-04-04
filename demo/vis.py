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

cache_path = "/workspace/qwen-cache/model.layers.31"

features = [
    37510,
    4938,
    23286,
    16578,
    15857,
    12302,
    33867,
    16075,
    40616,
    3725,
    19438,
    27590,
    12196,
    6255,
    12721,
    34921,
    35864,
    15821,
    3846,
    18402,
    32769,
    26386,
    14655,
    29545,
    21639,
    20008,
    25020,
    36433,
    31985,
    28080,
    3485,
    17656,
    20048,
    23192,
    34693,
    33691,
    12904,
    21436,
    25289,
    35311,
    5531,
    37848,
    24182,
    33979,
    13309,
    33142,
    6038,
    32570,
    7419,
    24829,
]

feature_display = make_feature_display(cache_path, features)
