# %%
# from autointerp import quantile_sampler, load
# from autointerp.automation import LogProbsClient, simulate
# from functools import partial
# import torch as t
# sampler = partial(quantile_sampler, n_quantiles=5, n_examples=5)


# client = LogProbsClient("Qwen/Qwen2.5-7B-Instruct", dtype=t.float16)

# features = load("/root/autointerp/cache/model.layers.0.pt", ctx_len=16, sampler=sampler, train=False, load_non_activating_test=False)

# explanation = "The word 'bleed'"
# simulate("explanation", features[0].activating_test_examples, client)

# %%

from autointerp.automation.clients import LocalClient
import asyncio
client = LocalClient("Qwen/Qwen2.5-7B-Instruct")

async def main():
    response = await client.generate(
        messages=[
            {"role": "user", "content": "Hello, how are you?"}
        ]
    )
    print(response) 

if __name__ == "__main__":
    asyncio.run(main())
