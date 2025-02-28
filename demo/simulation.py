# %%
from autointerp import quantile_sampler, load
from autointerp.automation import LogProbsClient, simulate
from functools import partial

sampler = partial(quantile_sampler, n_quantiles=5, n_examples=5)


client = LogProbsClient("Qwen/Qwen2.5-7B-Instruct")

features = load("/root/autointerp/cache/model.layers.0.pt", sampler=sampler, load_non_activating_test=False)

explanation = "The word 'bleed'"
simulate("explanation", features[0].activating_test_examples, client)







