from torch.utils.data import Dataset

from typing import Dict, List, Tuple
from transformers import AutoTokenizer, AddedToken
from neurondb import load_torch

SEP = AddedToken("<|sep|>")
UNK = AddedToken("<|unk|>")

# No space before input bc <|sep|> is added there.
PROMPT = """## Description: {explanation}

## Input:{example}"""

def prepare_tokenizer(tokenizer: AutoTokenizer):
    additional_special_tokens = tokenizer.additional_special_tokens

    # Add SEP token for correct separation without folding
        # Add UNK token for tokenization differences
    additional_special_tokens.extend([SEP, UNK])
    tokenizer.add_special_tokens(
        {"additional_special_tokens": additional_special_tokens}
    )

    print("ADDED SPECIAL TOKENS")
    return tokenizer


class SimulatorDataset(Dataset):
    def __init__(
        self,
        explanations: Dict,
        scores: Dict,
        subject_tokenizer: AutoTokenizer,
        simulator_tokenizer: AutoTokenizer,
        activation_paths: Dict,
    ):
        if (
            "<|sep|>" not in simulator_tokenizer.additional_special_tokens
            or "<|unk|>" not in simulator_tokenizer.additional_special_tokens
        ):
            raise ValueError("Must prepare tokenizer first.")

        self.explanations = explanations
        self.scores = scores

        self.activation_paths = activation_paths
        
        self.subject_tokenizer = subject_tokenizer
        self.simulator_tokenizer = simulator_tokenizer

        self.data = []

        self._build_dataset()

    def _get_response(
        self, example_tokens: List[int], predicted_activations: List[int]
    ) -> Tuple[str, str]:
        # Turn ids into str tokens
        example_str_tokens = self.subject_tokenizer.batch_decode(
            example_tokens
        )

        response_tokens = ["<|sep|>"]
        response_predictions = ["<|unk|>"]

        for i, (token, activation) in enumerate(
            zip(example_str_tokens, predicted_activations)
        ):
            # Get the tokenization of the str token by the simulator tokenizer
            simulator_tokens = self.simulator_tokenizer.encode(
                token, add_special_tokens=False
            )
            simulator_str_tokens = self.simulator_tokenizer.batch_decode(
                simulator_tokens
            )

            if len(simulator_str_tokens) > 1:
                print(simulator_str_tokens)

            # Add all tokens but the last, setting a trash prediction
            for non_last_token in simulator_str_tokens[:-1]:
                response_tokens.append(non_last_token)
                response_predictions.append("<|unk|>")

            # Add the last token and its prediction
            response_tokens.append(simulator_str_tokens[-1])
            activation_str = str(activation)
            response_predictions.append(activation_str)

            # Add a separator between tokens
            if i != len(example_str_tokens) - 1:
                response_tokens.append("<|sep|>")
                response_predictions.append("<|unk|>")

        formatted_response = "".join(response_tokens)
        formatted_labels = "".join(response_predictions)

        return formatted_response, formatted_labels, len(response_tokens)

    def _build_dataset(self):
        for layer, feature_index_and_explanation in self.explanations.items():
            # Get the activation path to load features from
            activation_path = self.activation_paths[layer]

            # Load all features, using same ctx len as explanation/simulation
            features = {
                f.index: f
                for f in load_torch(
                    activation_path, max_examples=2_000, ctx_len=32
                )
            }

            for (
                feature_index,
                explanation,
            ) in feature_index_and_explanation.items():
                # Load all the rounded EV predictions
                _, _, example_predictions = self.scores[layer][feature_index]

                # Get the examples for this feature index if they exist
                feature = features.get(int(feature_index), False)
                if not feature:
                    print(f"Feature {feature_index} not found")
                    continue
                examples = feature.examples

                for example, predicted_activations in zip(
                    examples, example_predictions
                ):
                    # Get the `List[int]` of true activations
                    true_activations = example.activations.tolist()

                    # Get the formatted response and labels
                    response, labels, response_len = self._get_response(
                        example.tokens, predicted_activations
                    )

                    prompt = PROMPT.format(
                        explanation=explanation,
                        example=response,
                    )

                    self.data.append(
                        {
                            "feature_index": feature_index,
                            "layer": layer,
                            "prompt": prompt,
                            "true_activations": true_activations,
                            "predicted_activations": predicted_activations,
                            "labels": labels,
                            "labels_start": -response_len,
                        }
                    )

            break

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
