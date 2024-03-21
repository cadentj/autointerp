from dataclasses import dataclass
from typing import List

import torch
from torch import Tensor

from rich.console import Console
from rich.table import Table

@dataclass 
class Location:
    feature_type: str # mlp, resid_in, or attn
    index: int # dictionary index
    layer: int # layer index

@dataclass
class Feature:
    prompt: str
    tokens: List[str]
    acts: Tensor
    n_acts: Tensor = None
    location: Location = None

@dataclass()
class State:
    agent: List
    self_reflector: List
    evaluator: dict

def log_conversation(
        conversation: List[dict], 
        log_file_path: str
    ) -> None:
    """Logs a conversation to a specified log file using rich for formatting.
    The conversation is expected to be a list of dictionaries with 'role' and 'content' keys.

    Args:
        conversation (List[dict]): A list of dictionaries, each containing 'role' and 'content'
        log_file_path (str): Path to the log file

    Returns:
        None
    """
    # Create a console object for outputting to the file
    console = Console(record=True, width=120)

    # Create a table with two columns: Role and Content
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Role", style="bold cyan")
    table.add_column("Content", justify="left")

    # Add each message to the table
    for message in conversation:
        role = message["role"].capitalize()  # Capitalize the role for better appearance
        content = message["content"]
        table.add_row(role, content)

    # Print the table to the console (which is recording)
    console.print(table)

    # Save the output to a log file
    console.save_text(log_file_path)

def gen(
        model, 
        messages: List[dict], 
        remote: bool = False, 
        max_new_tokens : int =300
    ) -> str:
    """Generate some tokens with nnsight and return new tokens.

    Args:
        model (LanguageModel): NNsight LanguageModel 
        messages (List[str]): Conversation history
        remote (bool, optional): Whether to use NDIF. Defaults to False.
        max_new_tokens (int, optional): How many new tokens to generate. Defaults to 300.
    
    Returns:
        str: New tokens.
    """
    
    prompt = model.tokenizer.apply_chat_template(messages, return_tensors="pt")

    sampling_kwargs = {
        "do_sample": True,
        "top_p": 0.3,
        "repetition_penalty": 1.1,
    }
        
    with model.generate(prompt, max_new_tokens=max_new_tokens, remote=remote, scan=False, validate=False, **sampling_kwargs):
        tokens = model.generator.output.save()
    
    new_tokens = tokens[0][len(prompt[0]):]
    torch.cuda.empty_cache()
    return model.tokenizer.decode(new_tokens)

def normalize_acts(acts: Tensor) -> Tensor:
    """Normalize activations to a scale of 0 to 10.

    Args:
        acts (Tensor): Activations to normalize

    Returns:
        Tensor: Normalized activations
    """

    return (acts - acts.min()) / (acts.max() - acts.min()) * 10

