from typing import List

from sentence_transformers import SentenceTransformer
import torch as t
from torchtyping import TensorType
from transformers import AutoTokenizer

from .schema import Example, Feature


def _normalize(
    activations: TensorType["seq"],
    max_activation: float,
) -> TensorType["seq"]:
    normalized = activations / max_activation * 10
    return normalized.round().int()


# TODO: Make this configurable
PAD_TOKEN_ID = 0
print("WARNING: USING PRESET PAD TOKEN ID (0)")


def quantile_sampler(
    token_windows: TensorType["batch", "seq"],
    activation_windows: TensorType["batch", "seq"],
    tokenizer: AutoTokenizer,
    n: int = 20,
    n_quantiles: int = 5,
):
    if len(token_windows) == 0:
        return None

    max_activation = activation_windows.max()
    examples_per_quantile = n // n_quantiles

    examples = []
    for i in range(n_quantiles):
        start_idx = i * examples_per_quantile
        end_idx = start_idx + examples_per_quantile

        for j in range(start_idx, end_idx):
            pad_token_mask = token_windows[j] == PAD_TOKEN_ID
            trimmed_window = token_windows[j][~pad_token_mask]
            trimmed_activation = activation_windows[j][~pad_token_mask]

            examples.append(
                Example(
                    trimmed_window,
                    trimmed_activation,
                    _normalize(trimmed_activation, max_activation),
                    quantile=n_quantiles - i,
                    str_tokens=tokenizer.batch_decode(trimmed_window),
                )
            )

    return examples


def max_activation_sampler(
    token_windows: TensorType["batch", "seq"],
    activation_windows: TensorType["batch", "seq"],
    tokenizer: AutoTokenizer,
    k: int = 20,
):
    if len(token_windows) < k:
        return None

    max_activation = activation_windows.max()
    examples = []
    for i in range(k):
        pad_token_mask = token_windows[i] == PAD_TOKEN_ID
        trimmed_window = token_windows[i][~pad_token_mask]
        trimmed_activation = activation_windows[i][~pad_token_mask]

        examples.append(
            Example(
                trimmed_window,
                trimmed_activation,
                _normalize(trimmed_activation, max_activation),
                quantile=-1,
                str_tokens=tokenizer.batch_decode(trimmed_window),
            )
        )

    return examples


class SimilaritySearch:
    def __init__(self, model_id: str, subject_id: str):
        self.model = SentenceTransformer(model_id)
        self.subject_tokenizer = AutoTokenizer.from_pretrained(subject_id)

        self.all_strides: TensorType["n_contexts", "seq_len"] = None
        self.valid_strides: TensorType["n_contexts", "seq_len"] = None
        self.embeddings: TensorType["n_contexts", "d_model"] = None

        self.loaded = False

    def load(
        self,
        tokens: TensorType["batch", "seq_len"],
        ctx_len: int,
        ignore_index: int,
    ):
        # Mask over invalid tokens
        mask = tokens != ignore_index
        mask_strides = mask.unfold(dimension=1, size=ctx_len, step=ctx_len)
        strides = tokens.unfold(dimension=1, size=ctx_len, step=ctx_len)

        # Filter for valid strides
        valid_strides_mask = mask_strides.all(dim=2)
        valid_strides = strides[valid_strides_mask]

        # Embed context windows
        str_data = self.subject_tokenizer.batch_decode(valid_strides)
        print("Encoding strides...")
        embeddings = self.model.encode(
            str_data, device="cuda", show_progress_bar=True
        )

        self.all_strides = strides
        self.valid_strides = valid_strides
        self.embeddings = t.from_numpy(embeddings)

    def _query(
        self,
        query_embeddings: TensorType["n_queries", "d_model"],
        locations: List[TensorType["features", 3]],
        k: int = 10,
    ):
        query_embeddings_normalized = query_embeddings / query_embeddings.norm(
            dim=1, keepdim=True
        )
        embeddings_normalized = self.embeddings / self.embeddings.norm(
            dim=1, keepdim=True
        )

        similarities: TensorType["n_queries", "n_contexts"] = t.matmul(
            query_embeddings_normalized, embeddings_normalized.t()
        )

        # set similarities to -inf for the stride indices
        for i, location in enumerate(locations):
            stride_indices = location[:, 0] * location[:, 1]
            similarities[i, stride_indices] = -float("inf")

        indices, _ = t.topk(similarities, k=k, dim=1)

        return indices

    def _get_similar_examples(
        self,
        indices: TensorType["k"],
    ):
        examples = []
        for index in indices:
            token_window = self.all_strides[index]
            pad_token_mask = token_window == PAD_TOKEN_ID
            trimmed_window = token_window[~pad_token_mask]
            activation_window = t.zeros_like(trimmed_window)

            examples.append(
                Example(
                    trimmed_window,
                    activation_window,
                    activation_window,
                    quantile=None,
                )
            )

        return examples

    def __call__(
        self,
        features: List[Feature],
        locations: TensorType["features", 3],
        batch_size: int = 64,
        k: int = 10,
    ):
        # Convert locations to stride indices
        ctx_len = self.all_strides.shape[1]
        locations = locations[:, 1] // ctx_len

        queries = [
            t.cat([e.tokens for e in feature.examples]) for feature in features
        ]
        queries = self.subject_tokenizer.batch_decode(queries)

        query_embeddings = self.model.encode(
            queries,
            device="cuda",
            show_progress_bar=True,
            convert_to_tensor=True,
        )

        query_embedding_batches = [
            (
                features[i : i + batch_size],
                query_embeddings[i : i + batch_size],
            )
            for i in range(0, len(query_embeddings), batch_size)
        ]

        for features, query_batch in query_embedding_batches:
            _locations = [
                locations[locations[:, 2] == f.index] for f in features
            ]
            topk_indices = self._query(query_batch, _locations, k=k)
            for i, feature in enumerate(features):
                feature.similar_examples = self._get_similar_examples(
                    topk_indices[i]
                )


def default_sampler(
    token_windows: TensorType["batch", "seq"],
    activation_windows: TensorType["batch", "seq"],
    tokenizer: AutoTokenizer,
    n_train: int = 20,
    n_test: int = 5,
    n_quantiles: int = 5,
    train: bool = True,
):
    if len(token_windows) < n_train + n_test:
        return None

    if train:
        return max_activation_sampler(
            token_windows[:n_train],
            activation_windows[:n_train],
            tokenizer,
            k=n_train,
        )
    else:
        return quantile_sampler(
            token_windows[n_train:],
            activation_windows[n_train:],
            tokenizer,
            n=n_test,
            n_quantiles=n_quantiles,
        )
