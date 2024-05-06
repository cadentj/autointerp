import torch as t
import openai
import replicate
from rich.console import Console
from rich.table import Table
from transformers import AutoTokenizer

from config import PROVIDER


if PROVIDER == "openai":
    print("Using OpenAI API")
    client = openai.OpenAI(api_key="<YOUR KEY HERE>")


def gen(prompt, generation_kwargs={}):
    if PROVIDER == "openai":
        return gen_openai(prompt, generation_kwargs)
    elif PROVIDER == "replicate":
        return gen_replicate(prompt, generation_kwargs)
    

def gen_openai(prompt, generation_kwargs={}):
    output = client.chat.completions.create(
        model="gpt-4-turbo-2024-04-09",
        messages=prompt,
        temperature = generation_kwargs.get("temperature", 1.0),
        max_tokens = generation_kwargs.get("max_tokens", 1000),
    )

    return output.choices[0].message.content


def gen_replicate(prompt, generation_kwargs={}):
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-70B-Instruct")

    prompt_str = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)

    query = {
        "prompt_template": prompt_str,
        **generation_kwargs
    }

    output = replicate.run(
        "meta/meta-llama-3-70b-instruct",
        input=query
    )

    return output


def unravel_index(flat_index, shape):

    indices = []
    for dim_size in reversed(shape):
        indices.append(flat_index % dim_size)
        flat_index = flat_index // dim_size
    return tuple(reversed(indices))


def topk(tensor, k):

    flat_tensor = tensor.flatten()

    top_values, flat_indices = t.topk(flat_tensor, k)

    original_indices = [unravel_index(idx.item(), tensor.size()) for idx in flat_indices]

    return top_values.tolist(), original_indices


def log(
    obj,
    role: str,
    message: str
):
    obj.state.history.append({"role": role, "message": message})


def log_conversation(
        conversation, 
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
    for i in range(len(conversation)):
        message = conversation[i]

        role = message["role"].capitalize()  # Capitalize the role for better appearance
        content = message["message"]
        if isinstance(content, list):
            content = ""
            for inner in message["message"]:
                content += "\n".join(inner)

        if message["role"].lower() == "section":
            table.add_row(role, content, style="bold", end_section=True)
        elif i != len(conversation) - 1 and conversation[i + 1]["role"].lower() == "section":
            table.add_row(role, content, end_section=True)
        else:
            table.add_row(role, content)

    # Print the table to the console (which is recording)
    console.print(table)

    # Save the output to a log file
    console.save_text(log_file_path)
