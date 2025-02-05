# Neuron DB

## Start an explainer vLLM server.

deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
meta-llama/Llama-3.3-70B-Instruct
deepseek-ai/DeepSeek-R1-Distill-Llama-8B
anthropic/claude-3.5-sonnet

`vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --enable-prefix-caching --dtype bfloat16 --disable-log-stats --gpu-memory-utilization 0.5`