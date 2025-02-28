import os

from transformers import AutoTokenizer

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

TEMPLATES = {
    "Qwen/Qwen2.5-7B-Instruct" : os.path.join(CURRENT_DIR, "tokenizer_chat_templates/qwen.jinja"),
    "Qwen/Qwen2.5-14B-Instruct" : os.path.join(CURRENT_DIR, "tokenizer_chat_templates/qwen.jinja"),
    "Qwen/Qwen2.5-1.5B-Instruct" : os.path.join(CURRENT_DIR, "tokenizer_chat_templates/qwen.jinja"),
    "Qwen/Qwen2.5-32B-Instruct" : os.path.join(CURRENT_DIR, "tokenizer_chat_templates/qwen.jinja")
}

def load_tokenizer(model_id: str) -> AutoTokenizer:
    """Load custom chat template for simulation scoring."""

    assert model_id in TEMPLATES, f"Model {model_id} not supported"

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    template_path = TEMPLATES[model_id]

    with open(template_path, "r") as f:
        template = f.read()

    tokenizer.chat_template = template

    return tokenizer