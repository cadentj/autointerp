# %%
import json
import os
from neurondb.autointerp.tune import (
    SimulatorDataset,
    prepare_tokenizer,
    SIMULATOR_PROMPT,
)
from transformers import AutoTokenizer

import torch as t
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM

from unsloth import FastLanguageModel

GEMMA_CACHE = "/share/u/caden/neurondb/cache/gemma-2-2b"
SIMULATOR_MODEL = "Qwen/Qwen2.5-3B"


def load_raw_data():
    with open(os.path.join(GEMMA_CACHE, "explanations.json"), "r") as f:
        explanations = json.load(f)

    scores = {}
    for file in os.listdir(GEMMA_CACHE):
        if "-scores.json" in file:
            with open(os.path.join(GEMMA_CACHE, file), "r") as f:
                raw_scores = json.load(f)

            for layer, layer_scores in raw_scores.items():
                scores[layer] = layer_scores

    path_template = (
        "/share/u/caden/neurondb/cache/gemma-2-2b/.model.layers.{layer}.pt"
    )
    activation_paths = {
        layer: path_template.format(layer=layer) for layer in scores.keys()
    }

    return explanations, scores, activation_paths


def load(simulator_tokenizer):
    explanations, scores, activation_paths = load_raw_data()

    subject_tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
    simulator_tokenizer = prepare_tokenizer(simulator_tokenizer)

    dataset = SimulatorDataset(
        explanations=explanations,
        scores=scores,
        subject_tokenizer=subject_tokenizer,
        simulator_tokenizer=simulator_tokenizer,
        activation_paths=activation_paths,
    )

    return dataset, simulator_tokenizer


def compute_loss_func(outputs, labels, num_items_in_batch: int = None):
    loss = t.nn.functional.cross_entropy(outputs, labels)
    return loss


def load_unsloth():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=SIMULATOR_MODEL,
        dtype=t.bfloat16,
        load_in_4bit=False,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=32,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,  # Supports any, but = 0 is optimized
        bias="none",  # Supports any, but = "none" is optimized
        use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
        random_state=3407,
    )
    return model, tokenizer


def train():
    model, simulator_tokenizer = load_unsloth()
    dataset, simulator_tokenizer = load(simulator_tokenizer)


    # def formatting_prompts_func(rows):
    #     print(rows)

    #     explanations = rows["explanation"]
    #     examples = rows["example"]
    #     texts = []

    #     for explanation, example in zip(explanations, examples):
    #         text = SIMULATOR_PROMPT.format(
    #             explanation=explanation, example=example
    #         )
    #         text += eos_token
    #         texts.append(text)
    #     return texts

    response_template = "## Input:"
    collator = DataCollatorForCompletionOnlyLM(
        response_template, tokenizer=simulator_tokenizer
    )

    config = SFTConfig(output_dir="/tmp")

    trainer = SFTTrainer(
        model,
        train_dataset=dataset,
        args=config,
        # formatting_func=formatting_prompts_func,
        data_collator=collator,
    )

    trainer.train()


if __name__ == "__main__":
    train()

# # %%

# from torch.utils.data import DataLoader

# # def collate_fn(batch):
# #     print(batch[0])
# #     return batch

# loader = DataLoader(dataset, batch_size=10, shuffle=True)

# batch = next(iter(loader))
# # %%
# batch['true_activations']
