# %%

qwen_1p5 = {
    # "2b_7b": "/root/neurondb/cache/gemma-2-2b-w65k-l0116-layer18/qwen-2.5-7b-instruct-simulated-by-Qwen2.5-1.5B-Instruct.json",
    "2b_70b": "/root/neurondb/cache/gemma-2-2b-w65k-l0116-layer18/llama-3.3-70b-instruct-simulated-by-Qwen2.5-1.5B-Instruct.json",
    # "9b_7b": "/root/neurondb/cache/gemma-2-9b-w131k-l098-layer28/qwen-2.5-7b-instruct-simulated-by-Qwen2.5-1.5B-Instruct.json",
    # "9b_70b": "/root/neurondb/cache/gemma-2-9b-w131k-l098-layer28/llama-3.3-70b-instruct-simulated-by-Qwen2.5-1.5B-Instruct.json",
}

qwen_7b = {
    # "2b_7b": "/root/neurondb/cache/gemma-2-2b-w65k-l0116-layer18/qwen-2.5-7b-instruct-simulated-by-Qwen2.5-7B-Instruct.json",
    "2b_70b": "/root/neurondb/cache/gemma-2-2b-w65k-l0116-layer18/llama-3.3-70b-instruct-simulated-by-Qwen2.5-7B-Instruct.json",
    # "9b_7b": "/root/neurondb/cache/gemma-2-9b-w131k-l098-layer28/qwen-2.5-7b-instruct-simulated-by-Qwen2.5-7B-Instruct.json",
    # "9b_70b": "/root/neurondb/cache/gemma-2-9b-w131k-l098-layer28/llama-3.3-70b-instruct-simulated-by-Qwen2.5-7B-Instruct.json",
}

qwen_14b = {
    # "2b_7b": "/root/neurondb/cache/gemma-2-2b-w65k-l0116-layer18/qwen-2.5-7b-instruct-simulated-by-Qwen2.5-14B-Instruct.json",
    "2b_70b": "/root/neurondb/cache/gemma-2-2b-w65k-l0116-layer18/llama-3.3-70b-instruct-simulated-by-Qwen2.5-14B-Instruct.json",
    # "9b_7b": "/root/neurondb/cache/gemma-2-9b-w131k-l098-layer28/qwen-2.5-7b-instruct-simulated-by-Qwen2.5-14B-Instruct.json",
    # "9b_70b": "/root/neurondb/cache/gemma-2-9b-w131k-l098-layer28/llama-3.3-70b-instruct-simulated-by-Qwen2.5-14B-Instruct.json",
}

qwen_32b = {
    # "2b_7b": "/root/neurondb/cache/gemma-2-2b-w65k-l0116-layer18/qwen-2.5-7b-instruct-simulated-by-Qwen2.5-32B-Instruct.json",
    "2b_70b": "/root/neurondb/cache/gemma-2-2b-w65k-l0116-layer18/llama-3.3-70b-instruct-simulated-by-Qwen2.5-32B-Instruct.json",
    # "9b_7b": "/root/neurondb/cache/gemma-2-9b-w131k-l098-layer28/qwen-2.5-7b-instruct-simulated-by-Qwen2.5-32B-Instruct.json",
    # "9b_70b": "/root/neurondb/cache/gemma-2-9b-w131k-l098-layer28/llama-3.3-70b-instruct-simulated-by-Qwen2.5-32B-Instruct.json",
}

import json
from collections import defaultdict

def get_ev_per_quantile(store, result, name):
    for _, scores in result.items():
        per_quantile = scores[0]
        for i, ev in enumerate(per_quantile):
            store[name][i].append(ev)

def load(paths, data, store):
    for name, path in paths.items():
        with open(path, "r") as f:
            results = json.load(f)
            data[name] = results

            get_ev_per_quantile(store, results, name)

            store[name] = {
                quantile: sum(evs) / len(evs)
                for quantile, evs in store[name].items()
            }

def get_top_features(data, model_size):
    top_features = sorted(
        [(feature, scores[1]) for feature, scores in data[model_size].items()],
        key=lambda x: x[1],
        reverse=True,
    )[:10]
    return top_features

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

def load_all_data():
    data_configs = {
        'Qwen 1.5B': (qwen_1p5, defaultdict(lambda: defaultdict(list)), {}),
        'Qwen 7B': (qwen_7b, defaultdict(lambda: defaultdict(list)), {}),
        'Qwen 14B': (qwen_14b, defaultdict(lambda: defaultdict(list)), {}),
        'Qwen 32B': (qwen_32b, defaultdict(lambda: defaultdict(list)), {})
    }
    
    for model_name, (paths, store, data) in data_configs.items():
        load(paths, data, store)
        
    return data_configs

def plot_combined_ev_per_quantile(data_configs):
    plt.figure(figsize=(12, 6))
    
    colors = {'2b_7b': '#1f77b4', '2b_70b': '#ff7f0e', 
              '9b_7b': '#2ca02c', '9b_70b': '#d62728'}
    
    line_styles = {
        'Qwen 1.5B': ':', 'Qwen 7B': '--', 
        'Qwen 14B': '-', 'Qwen 32B': '-.'
    }
    
    for model_name, (_, store, _) in data_configs.items():
        for config_name, evs in store.items():
            if len(evs) == 0:  # Skip if no data
                continue
            plt.plot(list(evs.values()), 
                    label=f'{model_name} - {config_name}',
                    color=colors[config_name],
                    linestyle=line_styles[model_name])

    plt.xticks(range(5))
    plt.xlabel("Quantile")
    plt.ylabel("EV Correlation")
    plt.title("Correlation per Quantile Across Models")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_combined_density(data_configs):
    plt.figure(figsize=(12, 6))
    
    colors = {'2b_7b': '#1f77b4', '2b_70b': '#ff7f0e', 
              '9b_7b': '#2ca02c', '9b_70b': '#d62728'}
    
    line_styles = {
        'Qwen 1.5B': ':', 'Qwen 7B': '--', 
        'Qwen 14B': '-', 'Qwen 32B': '-.'
    }
    
    for model_name, (_, _, data) in data_configs.items():
        if not data:  # Skip if no data
            continue
        for config_name, results in data.items():
            # Change here for different quantiles
            scores = [scores[1] for scores in results.values()]
            if not scores:  # Skip if no scores
                continue
                
            kde = gaussian_kde(scores)
            x_range = np.linspace(min(scores), max(scores), 200)
            density = kde(x_range)
            
            plt.plot(x_range, density, 
                    label=f'{model_name} - {config_name}',
                    color=colors[config_name],
                    linestyle=line_styles[model_name])

    plt.xlabel("Score")
    plt.ylabel("Density")
    plt.title("Score Distribution Across Models")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.grid(True, alpha=0.3)
    plt.show()

# Load all data and create plots
data_configs = load_all_data()
plot_combined_ev_per_quantile(data_configs)
plot_combined_density(data_configs)

