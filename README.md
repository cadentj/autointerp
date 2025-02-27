# autointerp

Simple automated interpretability. All other metrics are doomed so just use what works.

Cache activations:

```python
from autointerp import cache_activations

submodule = model.model.layers[0]
dictionary = ...

cache = cache_activations(
    model,
    {submodule : dictionary},
    tokens,
    max_tokens=5_000_000,
    batch_size=16,
)

cache.save_to_disk(
    save_dir,
    token_save_path
)
```

Then load, explain, and score features:

```python
from autointerp import load
from autointerp.automate import Explainer, Classifier, OpenRouterClient

client = OpenRouterClient(model=...)

explain = Explainer(client=client)
score = Classifier(client=client, method="detection")

# pretend this is in an async function
for f in load(save_dir, train=True): 
    explanation = await explain(f)
    score = await score(explanation, f)
```
