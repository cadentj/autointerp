from typing import List, Callable

from sentence_transformers import SentenceTransformer
import torch as t
import torch.nn.functional as F
from torchtyping import TensorType
from transformers import AutoTokenizer
from tqdm import tqdm

from .base import Example, Feature, NonActivatingType


def _normalize(
    activations: TensorType["seq"],
    max_activation: float,
) -> TensorType["seq"]:
    normalized = activations / max_activation * 10
    return normalized.round().int()


def identity_sampler(
    token_windows: TensorType["batch", "seq"],
    activation_windows: TensorType["batch", "seq"],
    tokenizer: AutoTokenizer,
) -> List[Example]:
    examples = []

    max_activation = activation_windows.max()

    for i in range(token_windows.shape[0]):
        pad_token_mask = token_windows[i] == tokenizer.pad_token_id
        trimmed_window = token_windows[i][~pad_token_mask]
        trimmed_activations = activation_windows[i][~pad_token_mask]

        examples.append(
            Example(
                tokens=trimmed_window,
                activations=trimmed_activations,
                normalized_activations=_normalize(
                    trimmed_activations, max_activation
                ),
                quantile=0,
                str_tokens=tokenizer.batch_decode(trimmed_window),
            )
        )

    return examples

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
        n_exclude: Number of examples to exclude from the beginning of each quantile.
        n_top_exclude: Number of top examples to exclude from the entire dataset.

    Returns:
        List of Example objects, with n_examples from each quantile.
        Returns None if there aren't enough examples to satisfy the requirements.
    """
    if n_top_exclude > 0:
        token_windows = token_windows[n_top_exclude:]
        activation_windows = activation_windows[n_top_exclude:]

    total_examples = token_windows.shape[0]
    quantile_size = total_examples // n_quantiles

    if (quantile_size - n_exclude) < n_examples:
        return None

    max_activation = activation_windows.max()

    examples = []
    for i in range(n_quantiles):
        quantile_start = i * quantile_size
        quantile_end = (
            (i + 1) * quantile_size if i < n_quantiles - 1 else total_examples
        )

        sample_start = quantile_start + n_exclude

        assert sample_start + n_examples <= quantile_end

        for j in range(sample_start, sample_start + n_examples):
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
                    # Reverse order so highest quantile is n_quantiles
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

    Args:
        n_examples: Number of examples to sample from each quantile.
        n_quantiles: Number of quantiles to sample from.
        n_exclude: Number of examples to exclude from the beginning of each quantile.
        n_top_exclude: Number of top examples to exclude from the entire dataset.

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


def _create_strides_and_ctx_locations(
    tokens: TensorType["batch", "seq_len"],
    locations: TensorType["batch", "2"],
    ctx_len: int,
):
    """Create strides and context locations from tokens and locations.

    Strides is the tokens reshaped into chunks of size ctx_len.
    Context locations is a tensor the length of locations. In each row, the first index
    is a stride index and the second index is a feature.
    """
    # Set context locations
    seq_len = tokens.shape[1]
    flat_indices = locations[:, 0] * seq_len + locations[:, 1]
    ctx_indices = flat_indices // ctx_len
    ctx_locations = t.stack([ctx_indices, locations[:, 2]], dim=1)
    max_ctx_idx = ctx_locations[:, 0].max().item()

    # Reshape strides and strides mask
    batch_size, seq_len = tokens.shape
    n_contexts = batch_size * (seq_len // ctx_len)
    strides = tokens.reshape(n_contexts, ctx_len)
    strides = strides[: max_ctx_idx + 1]

    return strides, ctx_locations


class RandomSampler:
    def __init__(
        self,
        subject_tokenizer: AutoTokenizer,
        tokens: TensorType["batch", "seq_len"],
        locations: TensorType["batch", "2"],
        ctx_len: int,
    ):
        self.subject_tokenizer = subject_tokenizer

        self.strides, self.ctx_locations = _create_strides_and_ctx_locations(
            tokens, locations, ctx_len
        )

    def __call__(self, features: List[Feature], n_examples: int = 10) -> None:
        all_random_idxs = t.rand(len(features), n_examples)

        for feature, random_idxs in tqdm(zip(features, all_random_idxs)):
            locations_mask = self.ctx_locations[:, 1] == feature.index
            locations_idxs = self.ctx_locations[locations_mask, 0]

            random_idxs = random_idxs * (locations_idxs.numel() - 1)
            random_idxs = random_idxs.round().int()

            non_activating_stride_idxs = locations_idxs[random_idxs]

            for stride_idx in non_activating_stride_idxs[:n_examples]:
                token_window = self.strides[stride_idx]
                pad_token_mask = (
                    token_window == self.subject_tokenizer.pad_token_id
                )
                trimmed_window = token_window[~pad_token_mask]
                activation_window = t.zeros_like(trimmed_window)

                feature.non_activating_examples.append(
                    Example(
                        tokens=trimmed_window,
                        activations=activation_window,
                        normalized_activations=activation_window,
                        quantile=NonActivatingType.RANDOM.value,  # Random non-activating
                        str_tokens=self.subject_tokenizer.batch_decode(
                            trimmed_window
                        ),
                    )
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
        subject_tokenizer: AutoTokenizer,
        tokens: TensorType["batch", "seq_len"],
        locations: TensorType["batch", "2"],
        ctx_len: int,
        embedding_model_id: str = "all-MiniLM-L6-v2",
    ):
        self.model = SentenceTransformer(embedding_model_id)
        self.subject_tokenizer = subject_tokenizer

        self.strides, self.ctx_locations = _create_strides_and_ctx_locations(
            tokens, locations, ctx_len
        )

        # Embed context windows
        str_data = self.subject_tokenizer.batch_decode(self.strides)
        print("Encoding token contexts...")
        embeddings = self.model.encode(
            str_data,
            device="cuda",
            show_progress_bar=True,
            convert_to_tensor=True,
        )

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
            pad_token_mask = token_window == self.subject_tokenizer.pad_token_id
            trimmed_window = token_window[~pad_token_mask]
            activation_window = t.zeros_like(trimmed_window)

            examples.append(
                Example(
                    tokens=trimmed_window,
                    activations=activation_window,
                    normalized_activations=activation_window,
                    quantile=NonActivatingType.SIMILAR.value,
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
        # Concatenate the first 10 examples of each feature into a single "query"
        queries = [
            t.cat([e.tokens for e in feature.activating_examples[:10]])
            for feature in features
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
                feature.non_activating_examples = self._get_similar_examples(
                    idxs
                )
