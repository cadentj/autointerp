import torch as t

from autointerp import make_quantile_sampler, load
from autointerp.automation import LogProbsClient, simulate

sampler = make_quantile_sampler(n_examples=5, n_quantiles=5, n_top_exclude=20)
client = LogProbsClient("Qwen/Qwen2.5-7B-Instruct", torch_dtype=t.bfloat16)

features = load(
    "/root/autointerp/cache/model.layers.0.pt",
    ctx_len=16,
    sampler=sampler,
)

simulate("some explanation here", features[0].activating_examples, client)
