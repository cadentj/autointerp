# %%
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

data = json.load(open("/share/u/caden/neurondb/experiments/crosscoders/detection/scores_data_0_1025.json"))

def calculate_accuracy(stats):
    """Calculate accuracy from confusion matrix stats"""
    tp = stats['TP']
    tn = stats['TN']
    fp = stats['FP']
    fn = stats['FN']
    
    total = tp + tn + fp + fn
    return (tp + tn) / total if total > 0 else 0

# Convert data to a format suitable for plotting
results = []
for feature, inner_dict in data.items():
    for quantile, stats in inner_dict.items():
        accuracy = calculate_accuracy(stats)
        results.append({
            'feature': feature,
            'quantile': quantile,
            'accuracy': accuracy
        })

df = pd.DataFrame(results)

# Create figure for average accuracy plot
fig, ax = plt.subplots(figsize=(10, 6))

# Calculate mean accuracy for each quantile
mean_accuracy = df.groupby('quantile')['accuracy'].mean()
quantiles = sorted(df['quantile'].unique())

# Plot average accuracy
ax.plot(quantiles, mean_accuracy, marker='o', linewidth=2)

# Add error bars (standard deviation)
std_accuracy = df.groupby('quantile')['accuracy'].std()
ax.fill_between(quantiles, 
                mean_accuracy - std_accuracy,
                mean_accuracy + std_accuracy,
                alpha=0.2)

# Customize plot
ax.set_title('Average Accuracy by Quantile')
ax.set_xlabel('Quantile')
ax.set_ylabel('Average Accuracy')
ax.set_ylim(0, 1)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %%


df[df['feature'] == '2']