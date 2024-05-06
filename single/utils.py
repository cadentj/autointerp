import torch as t
import replicate

def gen(prompt):
    output = replicate.run(
        "meta/meta-llama-3-70b-instruct",
        input=prompt
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

