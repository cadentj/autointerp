# %%
import copy
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache, StaticCache
from accelerate.test_utils.testing import get_backend

DEVICE, _, _ = get_backend() # automatically detects the underlying device type (CUDA, CPU, XPU, MPS, etc.)
model_id = "Qwen/Qwen2.5-1.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map=DEVICE)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Init StaticCache with big enough max-length (1024 tokens for the below example)
# You can also init a DynamicCache, if that suits you better
prompt_cache = StaticCache(config=model.config, max_batch_size=2, max_cache_len=1024, device=DEVICE, dtype=torch.bfloat16)

INITIAL_PROMPT = [{'role': 'system', 'content': "You are a helpful assistant. "}]
text = tokenizer.apply_chat_template([INITIAL_PROMPT] * 2, tokenize=False)
inputs_initial_prompt = tokenizer(text, return_tensors="pt").to(DEVICE)
print(inputs_initial_prompt['input_ids'].shape)
# This is the common prompt cached, we need to run forward without grad to be abel to copy
with torch.no_grad():
     prompt_cache = model(**inputs_initial_prompt, past_key_values = prompt_cache).past_key_values

prompts = ["Help me to write a blogpost about travelling.", "What is the capital of France?"]
responses = []
for prompt in prompts:
    new_inputs = tokenizer.apply_chat_template([INITIAL_PROMPT + [{'role': 'user', 'content': prompt}]]  *2, tokenize=False)
    new_inputs = tokenizer(new_inputs, return_tensors="pt").to(DEVICE)
    past_key_values = copy.deepcopy(prompt_cache)
    outputs = model(**new_inputs, past_key_values=past_key_values)
    responses.append(outputs)



# %%

responses[0].logits[:,-1,:].argmax(dim=-1)

# %%

tokenizer.decode(151644)