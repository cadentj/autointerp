# %%

save_path = "/root/autointerp/cache/model.layers.31.pt"

from autointerp import load, make_quantile_sampler

features = load(save_path, make_quantile_sampler(n_examples=10, n_quantiles=1))

# %%

idx = 8
print(features[idx].max_activation)
features[idx].display()