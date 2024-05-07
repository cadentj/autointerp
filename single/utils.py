import torch as t
import openai
import replicate
from rich.console import Console
from rich.table import Table
from transformers import AutoTokenizer

from config import PROVIDER
from keys import OAI

def gen(
        prompt, 
        postprocess, 
        generation_kwargs={},
        verbose=True
    ):
    if PROVIDER == "openai":
        return gen_openai(prompt, postprocess, generation_kwargs, verbose)
    elif PROVIDER == "replicate":
        return gen_replicate(prompt, postprocess, generation_kwargs, verbose)
    

def gen_openai(prompt, postprocess, generation_kwargs={}, verbose=True):
    client = openai.OpenAI(api_key=OAI)

    output = client.chat.completions.create(
        model="gpt-4-turbo-2024-04-09",
        messages=prompt,
        temperature = generation_kwargs.get("temperature", 1.0),
        max_tokens = generation_kwargs.get("max_tokens", 1000),
    )
    
    output = output.choices[0].message.content

    try:
        processed_output = postprocess(output)
    except Exception as e:
        print(f"Postprocessing failed: {e}")
        processed_output = "FAILED"

    if verbose:
        return processed_output, output

    return processed_output


def gen_replicate(prompt, postprocess, generation_kwargs={}, verbose=True):
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

    output_str = "".join(output)

    try:
        processed_output = postprocess(output_str)
    except:
        print("Postprocessing failed")
        processed_output = "FAILED"

    if verbose:
        return processed_output, output

    return processed_output


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
    state,
    role: str,
    message: str
):
    state.history.append({"role": role, "message": message})


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
