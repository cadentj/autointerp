# %%

import json
import matplotlib.pyplot as plt
import numpy as np

scores_path = "/share/u/caden/neurondb/cache/gemma-2-2b/scores.json"

with open(scores_path, "r") as f:
    raw_scores = json.load(f)["0"]

scores = [s[1] for s in raw_scores.values()]

# %%
# Create the density plot
plt.figure(figsize=(10, 6))
plt.hist(scores, bins=50, density=True, alpha=0.7)

# Add KDE curve
density = plt.gca().get_children()[0]
xs = np.linspace(min(scores), max(scores), 200)
bandwidth = 0.5 * np.std(scores) * len(scores)**(-0.2)  # Scott's rule
kernel = np.exp(-0.5 * ((xs[:, np.newaxis] - scores) / bandwidth)**2)
kde = np.mean(kernel, axis=1) / (bandwidth * np.sqrt(2 * np.pi))
plt.plot(xs, kde, 'r-', lw=2)

plt.title('Distribution of Scores')
plt.xlabel('Score')
plt.ylabel('Density')
plt.grid(True, alpha=0.3)
plt.show()

# %%
