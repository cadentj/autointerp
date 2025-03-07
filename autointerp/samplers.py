from typing import List, Callable

from sentence_transformers import SentenceTransformer
import torch as t
import torch.nn.functional as F
from torchtyping import TensorType
from transformers import AutoTokenizer

from .base import Example, Feature


def _normalize(
    activations: TensorType["seq"],
    max_activation: float,
) -> TensorType["seq"]:
    normalized = activations / max_activation * 10
    return normalized.round().int()


def quantile_sampler(
    token_windows: TensorType["batch", "seq"],
    activation_windows: TensorType["batch", "seq"],
    tokenizer: AutoTokenizer,
    n_examples: int,
    n_quantiles: int,
    n_exclude: int,
    n_top_exclude: int,
) -> List[Example]:
    """Sample the top n_examples from each quantile of the activation distribution.

    Sampling n_examples from 1 quantile is equivalent to max activation sampling.

    Args:
        token_windows: Tensor of shape (batch, seq) containing the tokens of the windows.
        activation_windows: Tensor of shape (batch, seq) containing the activations of the windows.
        tokenizer: Tokenizer to decode the windows.
        n_examples: Number of examples to sample from each quantile.
        n_quantiles: Number of quantiles to sample from.
        n_exclude: Number of top examples to exclude from sampling.
    """
    token_windows = token_windows[n_top_exclude:]
    activation_windows = activation_windows[n_top_exclude:]

    n_excluded = n_quantiles * n_exclude
    if len(token_windows) < (n_examples - n_excluded):
        return None

    max_activation = activation_windows.max()
    examples_per_quantile = n_examples // n_quantiles

    examples = []
    for i in range(n_quantiles):
        start_idx = (i * examples_per_quantile) + n_exclude
        end_idx = start_idx + examples_per_quantile

        for j in range(start_idx, end_idx):
            pad_token_mask = token_windows[j] == tokenizer.pad_token_id
            trimmed_window = token_windows[j][~pad_token_mask]
            trimmed_activation = activation_windows[j][~pad_token_mask]

            examples.append(
                Example(
                    tokens=trimmed_window,
                    activations=trimmed_activation,
                    normalized_activations=_normalize(
                        trimmed_activation, max_activation
                    ),
                    quantile=n_quantiles - i,
                    str_tokens=tokenizer.batch_decode(trimmed_window),
                )
            )

    return examples

def make_quantile_sampler(
    n_examples: int = 20,
    n_quantiles: int = 1,
    n_exclude: int = 0,
    n_top_exclude: int = 0,
) -> Callable:
    """Create a quantile sampler function.

    Sampling n_examples from 1 quantile is equivalent to max activation sampling.

    Args:
        n_examples: Number of examples to sample from each quantile.
        n_quantiles: Number of quantiles to sample from.
        n_exclude: Number of examples to exclude from sampling per quantile.
        n_top_exclude: Number of top examples to exclude from sampling.

    Returns:
        A function that samples examples from a quantile.
    """

    from functools import partial
    return partial(
        quantile_sampler,
        n_examples=n_examples,
        n_quantiles=n_quantiles,
        n_exclude=n_exclude,
        n_top_exclude=n_top_exclude,
    )

class SimilaritySearch:
    """Use similarity search to sample non-activating examples.

    Calling an instance of this class on a batch of features will initialize their
    non_activating_examples with examples sampled from the token dataset. 

    Args:
        subject_model_id: The model id of the subject model.
        tokens: The tokens of the subject model.
        locations: The locations of the subject model.
        ctx_len: The context length of the subject model.
        embedding_model_id: The model id of the embedding model.
    """

    def __init__(
        self,
        subject_model_id: str,
        tokens: TensorType["batch", "seq_len"],
        locations: TensorType["batch", "2"],
        ctx_len: int,
        embedding_model_id: str = "all-MiniLM-L6-v2",
    ):
        self.model = SentenceTransformer(embedding_model_id)
        self.subject_tokenizer = AutoTokenizer.from_pretrained(
            subject_model_id
        )

        # Set context locations
        seq_len = tokens.shape[1]
        flat_indices = locations[:, 0] * seq_len + locations[:, 1]
        ctx_indices = flat_indices // ctx_len
        self.ctx_locations = t.stack([ctx_indices, locations[:, 2]], dim=1)
        max_ctx_idx = self.ctx_locations[:, 0].max().item()

        # Reshape strides and strides mask
        batch_size, seq_len = tokens.shape
        n_contexts = batch_size * (seq_len // ctx_len)
        strides = tokens.reshape(n_contexts, ctx_len)
        strides = strides[: max_ctx_idx + 1]

        # Embed context windows
        str_data = self.subject_tokenizer.batch_decode(strides)
        print("Encoding token contexts...")
        embeddings = self.model.encode(
            str_data,
            device="cuda",
            show_progress_bar=True,
            convert_to_tensor=True,
        )

        # Set all strides and valid strides
        self.strides = strides

        # Normalize embeddings
        normalized_embeddings = F.normalize(embeddings, p=2, dim=1).t()
        self.normalized_embeddings = normalized_embeddings.to("cuda")

    def _query(
        self,
        features: List[Feature],
        query_embeddings: TensorType["n_queries", "d_model"],
        k: int = 10,
    ) -> TensorType["n_queries", "n_contexts"]:
        """Query the normalized embeddings to find the top k most similar contexts.

        Args:
            features: A list of features.
            query_embeddings: The query embeddings.
            k: The number of most similar contexts to return.
        """
        # Compute the similarity between query top examples
        query_embeddings_normalized = F.normalize(query_embeddings, p=2, dim=1)
        similarities: TensorType["n_queries", "n_contexts"] = t.matmul(
            query_embeddings_normalized, self.normalized_embeddings
        )

        for i, features in enumerate(features):
            idx = features.index

            # Get context indices where the feature is activating
            mask = self.ctx_locations[:, 1] == idx
            locations = self.ctx_locations[mask]

            # Set the similarity of the activating contexts to -inf
            similarities[i, locations[:, 0]] = -float("inf")

        _, indices = t.topk(similarities, k=k, dim=1)

        return indices.cpu()

    def _get_similar_examples(
        self,
        idxs: TensorType["k"],
    ) -> List[Example]:
        """Given context indices, create non-activating examples."""
        examples = []
        for idx in idxs:
            token_window = self.strides[idx]
            pad_token_mask = (
                token_window == self.subject_tokenizer.pad_token_id
            )
            trimmed_window = token_window[~pad_token_mask]
            activation_window = t.zeros_like(trimmed_window)

            examples.append(
                Example(
                    tokens=trimmed_window,
                    activations=activation_window,
                    normalized_activations=activation_window,
                    quantile=0,
                    str_tokens=self.subject_tokenizer.batch_decode(
                        trimmed_window
                    ),
                )
            )

        return examples

    def __call__(
        self,
        features: List[Feature],
        batch_size: int = 64,
        n_examples: int = 10,
    ) -> None:
        """Sample non-activating examples from the token dataset.

        Args:
            features: A list of features.
            batch_size: The batch size for encoding.
            n_examples: The number of examples to sample.
        """
        # Concatenate the first 20 examples of each feature into a single "query"
        queries = [
            t.cat([e.tokens for e in feature.examples[:20]]) for feature in features
        ]
        queries = self.subject_tokenizer.batch_decode(queries)

        # Encode the queries
        query_embeddings = self.model.encode(
            queries,
            device="cuda",
            show_progress_bar=True,
            convert_to_tensor=True,
        )

        # Batch the queries
        query_embedding_batches = [
            (
                features[i : i + batch_size],
                query_embeddings[i : i + batch_size],
            )
            for i in range(0, len(query_embeddings), batch_size)
        ]

        # Run similarity search for each batch and add non-activating examples to the features
        for features, query_batch in query_embedding_batches:
            topk_indices = self._query(features, query_batch, k=n_examples)
            for idxs, feature in zip(topk_indices, features):
                feature.non_activating_examples = (
                    self._get_similar_examples(idxs)
                )