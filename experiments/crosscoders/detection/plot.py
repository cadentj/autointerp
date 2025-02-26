# %%
import json
import pandas as pd
import matplotlib.pyplot as plt

detection = json.load(open("/share/u/caden/neurondb/experiments/crosscoders/detection/scores_data_0_1025.json"))
fuzzing = json.load(open("/share/u/caden/neurondb/experiments/crosscoders/fuzzing/scores_data_0_1025.json"))

def calculate_accuracy(stats):
    """Calculate accuracy from confusion matrix stats"""
    tp = stats['TP']
    tn = stats['TN']
    fp = stats['FP']
    fn = stats['FN']
    
    total = tp + tn + fp + fn
    return (tp + tn) / total if total > 0 else 0

def calculate_total_accuracy(inner_dict):
    acc = 0
    for quantile, stats in inner_dict.items():
        acc += calculate_accuracy(stats)
    return acc / len(inner_dict)

# Process detection data
detection_results = []
for feature, inner_dict in detection.items():
    detection_results.append({
        'feature': feature,
        'accuracy': calculate_total_accuracy(inner_dict)
    })

# Process fuzzing data
fuzzing_results = []
for feature, inner_dict in fuzzing.items():
    fuzzing_results.append({
        'feature': feature,
        'accuracy': calculate_total_accuracy(inner_dict)
    })

# Convert to DataFrames
detection_df = pd.DataFrame(detection_results)
fuzzing_df = pd.DataFrame(fuzzing_results)

# Merge the dataframes
merged_df = pd.merge(
    detection_df, 
    fuzzing_df, 
    on=['feature'], 
    suffixes=('_detection', '_fuzzing')
)

# Create scatter plot
plt.figure(figsize=(10, 8))
plt.scatter(merged_df['accuracy_detection'], 
           merged_df['accuracy_fuzzing'],
           alpha=0.6)

# Add diagonal line
min_val = min(merged_df['accuracy_detection'].min(), merged_df['accuracy_fuzzing'].min())
max_val = max(merged_df['accuracy_detection'].max(), merged_df['accuracy_fuzzing'].max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)

# Add labels and title
plt.xlabel('Detection Accuracy')
plt.ylabel('Fuzzing Accuracy')
plt.title('Detection vs Fuzzing Accuracy by Feature')

# Add grid
plt.grid(True, alpha=0.3)

# Make plot square
plt.axis('square')

# Set axis limits with a small margin
margin = 0.05
plt.xlim(min_val - margin, max_val + margin)
plt.ylim(min_val - margin, max_val + margin)

plt.show()




# %%



