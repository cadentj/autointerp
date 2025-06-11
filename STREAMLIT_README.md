# Autointerp Streamlit Dashboard

A comprehensive web interface for running autointerp feature explanation and classification tasks.

## Features

- **Feature Loading**: Load cached features from subdirectories
- **Explanation Tab**: Generate explanations for neural network features using customizable prompts
- **Classification Tab**: Classify feature examples using detection methods
- **Job Progress Tracking**: Real-time progress monitoring for long-running tasks
- **Client Support**: Compatible with local and OpenRouter API clients

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements-streamlit.txt
```

2. Ensure you have your autointerp environment set up with cached features.

## Usage

### Basic Launch

```bash
python run_dashboard.py /path/to/your/cached/features
```

### Advanced Launch Options

```bash
# Custom port and host
python run_dashboard.py /path/to/features --port 8502 --host 0.0.0.0

# Debug mode
python run_dashboard.py /path/to/features --debug
```

### Direct Streamlit Launch

You can also run directly with streamlit:

```bash
streamlit run app.py -- /path/to/your/cached/features
```

## Interface Layout

### Left Sidebar
- **Feature Loading**: Directory selection interface
- **Navigation**: Tab selection (Explain/Classify)

### Main Content Area
- **Explain Tab**: 
  - Editable explainer prompt with Python template strings
  - Sampler configuration (quantile, identity)
  - Client configuration (Local, OpenRouter)
  - Explainer options (max/min activation, threshold, etc.)
  
- **Classify Tab**:
  - Editable classification prompt
  - Classification options (batch size, thresholds)
  - Sampler configuration
  - Client configuration
  - Classifier options (examples shown, method)

### Right Sidebar
- **Running Jobs**: Real-time progress bars for active tasks
- **Job Status**: Completed, running, and error states

## Feature Directory Structure

The dashboard expects a directory structure like:
```
/path/to/features/
├── experiment_1/
│   ├── layer_0.pt
│   ├── layer_1.pt
│   └── ...
├── experiment_2/
│   ├── 0.pt
│   ├── 1.pt
│   ├── header.parquet
│   └── ...
└── ...
```

Each subdirectory should contain cached feature activations in the format expected by `autointerp.loader.load()`.

## Configuration

### Client Setup

**Local Client**: Assumes you have a local server running on `localhost:30000`

**OpenRouter Client**: Requires `OPENROUTER_KEY` environment variable:
```bash
export OPENROUTER_KEY="your-api-key-here"
```

### Sampling Options

- **Quantile Sampler**: Configure number of examples per quantile, number of quantiles, and exclusion parameters
- **Identity Sampler**: Uses examples as-is without additional sampling

## Workflow

1. **Load Features**: Select a feature directory and load cached features
2. **Run Explanations**: Configure prompts and options, then run explanation job
3. **Run Classification**: After explanations complete, run classification job
4. **Monitor Progress**: Track job progress in the right sidebar
5. **View Results**: Examine results in the expandable results section

## Advanced Features

### Custom Prompts
- Edit explainer and classification prompts directly in the interface
- Supports Python template string formatting
- Real-time prompt preview

### Async Job Management
- Jobs run in background threads to avoid blocking the UI
- Multiple jobs can run simultaneously
- Job progress is updated in real-time

### Result Export
- Results are displayed in JSON format
- Can be copied for further analysis

## Troubleshooting

### Common Issues

1. **Features not loading**: Ensure the directory path is correct and contains valid cached features
2. **Client errors**: Check your API keys and server configurations
3. **Job stuck**: Jobs may take significant time for large feature sets; check the progress bar
4. **Memory issues**: Large feature sets may require more RAM; consider reducing max_examples

### Debug Mode

Run with `--debug` flag for additional logging information.

## API Compatibility

This dashboard is compatible with:
- Local inference servers (llama.cpp, vllm, etc.)
- OpenRouter API
- Any OpenAI-compatible API endpoint

## Performance Notes

- Feature loading can be memory-intensive for large datasets
- Explanation and classification jobs are CPU/GPU intensive
- Consider running on machines with adequate resources for your feature set size