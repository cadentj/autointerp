from ...base import Example
from ..clients import Conversation
from typing import List, Dict
from transformers import AutoTokenizer

MAX_ACTIVATION = 9
UNKNOWN_ACTIVATION_STRING = "unknown"

PROMPT_TEMPLATE = """Neuron {index}
Explanation of neuron {index} behavior: {explanation}"""

ASSISTANT_TEMPLATE = """Activations:
<start>
{example}
<end>"""

SYSTEM_PROMPT = f"""We're studying neurons in a neural network.
Each neuron looks for some particular thing in a short document.
Look at summary of what the neuron does, and try to predict how it will fire on each token.

The activation format is token<tab>activation, activations go from 0 to {MAX_ACTIVATION}, "unknown" indicates an unknown activation. Most activations will be 0.
"""


def _normalize_activations(
    activations: List[float], max_activation: float
) -> List[int]:
    """Normalize activations to be between 0 and MAX_ACTIVATION."""
    return [
        int(activation / max_activation * MAX_ACTIVATION) for activation in activations
    ]


def _format_example(
    activation_record: Dict, start_index: int, max_activation: float
) -> str:
    """Format an example into a string of newline, tab separated tokens 
    and activations.

    Tokens before the start index are marked with "unknown" activation.
    """

    tokens = activation_record["tokens"]
    activations = activation_record["activations"]
    normalized_activations = _normalize_activations(
        activations, max_activation
    )

    formatted_example = []
    for i, (token, activation) in enumerate(
        zip(tokens, normalized_activations)
    ):
        if i < start_index:
            formatted_example.append(
                f"{token}\t{UNKNOWN_ACTIVATION_STRING}"
            )
        else:
            formatted_example.append(
                f"{token}\t{activation}"
            )

    return "\n".join(formatted_example)

def _format_example_for_simulation(example: Example, tokenizer: AutoTokenizer) -> str: 
    """Format an example into a string of newline, tab separated tokens 
    and activations.
    """

    str_tokens = tokenizer.batch_decode(example.tokens, skip_special_tokens=True)

    formatted_example = []
    for token in str_tokens:
        formatted_example.append(f"{token}\t{UNKNOWN_ACTIVATION_STRING}")

    return "\n".join(formatted_example)

def _get_max_activation(examples):
    """Get the maximum activation from a list of examples."""
    max_activation = 0.0
    for example in examples:
        max_activation = max(max_activation, max(example['activations']))
    return max_activation

def _build_messages(conversation: List[str]) -> List[Dict]:
    """Build a list of messages from a conversation."""
    messages = [{
        "role": "system",
        "content": conversation[0],
    }]

    for idx, message in enumerate(conversation[1:]):
        messages.append({
            "role": "user" if idx % 2 == 0 else "assistant",
            "content": message,
        })

    return messages

def format_prompt(explanation: str, example: Example, tokenizer: AutoTokenizer) -> Conversation:
    """Format a list of few shot examples into a prompt."""

    messages = [SYSTEM_PROMPT]

    for i, feature in enumerate(FEW_SHOT_EXAMPLES):
        max_activation = _get_max_activation(feature["activation_records"])

        formatted_examples = "\n<end>\n<start>\n".join(
            _format_example(activation_record, start_index, max_activation)
            for activation_record, start_index in zip(
                feature["activation_records"],
                feature["first_revealed_activation_indices"],
            )
        )

        index = i + 1
        user_prompt = PROMPT_TEMPLATE.format(
            index=index,
            explanation=feature["explanation"],
        )

        assistant_response = ASSISTANT_TEMPLATE.format(
            example=formatted_examples
        )

        messages.extend([
            user_prompt,
            assistant_response,
        ])

    user_prompt = PROMPT_TEMPLATE.format(
        explanation=explanation,
        index=i + 2,
    )

    assistant_response = ASSISTANT_TEMPLATE.format(
        example=_format_example_for_simulation(example, tokenizer)
    )

    messages.extend([
        user_prompt,
        assistant_response,
    ])

    return _build_messages(messages)


FEW_SHOT_EXAMPLES = [
    {
        "activation_records": [
            {
                "tokens": [
                    "The",
                    " editors",
                    " of",
                    " Bi",
                    "opol",
                    "ym",
                    "ers",
                    " are",
                    " delighted",
                    " to",
                    " present",
                    " the",
                    " ",
                    "201",
                    "8",
                    " Murray",
                    " Goodman",
                    " Memorial",
                    " Prize",
                    " to",
                    " Professor",
                    " David",
                    " N",
                    ".",
                    " Ber",
                    "atan",
                    " in",
                    " recognition",
                    " of",
                    " his",
                    " seminal",
                    " contributions",
                    " to",
                    " bi",
                    "oph",
                    "ysics",
                    " and",
                    " their",
                    " impact",
                    " on",
                    " our",
                    " understanding",
                    " of",
                    " charge",
                    " transport",
                    " in",
                    " biom",
                    "olecules",
                    ".\n\n",
                    "In",
                    "aug",
                    "ur",
                    "ated",
                    " in",
                    " ",
                    "200",
                    "7",
                    " in",
                    " honor",
                    " of",
                    " the",
                    " Bi",
                    "opol",
                    "ym",
                    "ers",
                    " Found",
                    "ing",
                    " Editor",
                    ",",
                    " the",
                    " prize",
                    " is",
                    " awarded",
                    " for",
                    " outstanding",
                    " accomplishments",
                ],
                "activations": [
                    0,
                    0.01,
                    0.01,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0.04,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    3.39,
                    0.12,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0.41,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
            }
        ],
        "first_revealed_activation_indices": [7],
        "explanation": "language related to something being groundbreaking",
    },
    {
        "activation_records": [
            {
                "tokens": [
                    '{"',
                    "widget",
                    "Class",
                    '":"',
                    "Variant",
                    "Matrix",
                    "Widget",
                    '","',
                    "back",
                    "order",
                    "Message",
                    '":"',
                    "Back",
                    "ordered",
                    '","',
                    "back",
                    "order",
                    "Message",
                    "Single",
                    "Variant",
                    '":"',
                    "This",
                    " item",
                    " is",
                    " back",
                    "ordered",
                    '.","',
                    "ordered",
                    "Selection",
                    '":',
                    "true",
                    ',"',
                    "product",
                    "Variant",
                    "Id",
                    '":',
                    "0",
                    ',"',
                    "variant",
                    "Id",
                    "Field",
                    '":"',
                    "product",
                    "196",
                    "39",
                    "_V",
                    "ariant",
                    "Id",
                    '","',
                    "back",
                    "order",
                    "To",
                    "Message",
                    "Single",
                    "Variant",
                    '":"',
                    "This",
                    " item",
                    " is",
                    " back",
                    "ordered",
                    " and",
                    " is",
                    " expected",
                    " by",
                    " {",
                    "0",
                    "}.",
                    '","',
                    "low",
                    "Price",
                    '":',
                    "999",
                    "9",
                    ".",
                    "0",
                    ',"',
                    "attribute",
                    "Indexes",
                    '":[',
                    '],"',
                    "productId",
                    '":',
                    "196",
                    "39",
                    ',"',
                    "price",
                    "V",
                    "ariance",
                    '":',
                    "true",
                    ',"',
                ],
                "activations": [
                    0,
                    0,
                    0,
                    0,
                    4.2,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    3.6,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    3.7,
                    0,
                    0,
                    0,
                    0,
                    4.02,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    3.5,
                    3.7,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    2.9,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    2.3,
                    2.24,
                    0,
                    0,
                    0,
                ],
            },
            {
                "tokens": [
                    "A",
                    " regular",
                    " look",
                    " at",
                    " the",
                    " ups",
                    " and",
                    " downs",
                    " of",
                    " variant",
                    " covers",
                    " in",
                    " the",
                    " comics",
                    " industry",
                    "…\n\n",
                    "Here",
                    " are",
                    " the",
                    " Lego",
                    " variant",
                    " sketch",
                    " covers",
                    " by",
                    " Leon",
                    "el",
                    " Cast",
                    "ell",
                    "ani",
                    " for",
                    " a",
                    " variety",
                    " of",
                    " Marvel",
                    " titles",
                    ",",
                ],
                "activations": [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    6.52,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    1.62,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    3.23,
                    0,
                    0,
                    0,
                    0,
                ],
            },
        ],
        "first_revealed_activation_indices": [2, 8],
        "explanation": "the word “variant” and other words with the same ”vari” root",
    },
]
