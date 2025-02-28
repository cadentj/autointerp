from typing import List

from sentence_transformers import SentenceTransformer
import torch as t
from torchtyping import TensorType
from transformers import AutoTokenizer

from .base import Example, Feature


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
                tokens=trimmed_window,
                activations=trimmed_activation,
                normalized_activations=_normalize(
                    trimmed_activation, max_activation
                ),
                quantile=-1,
                str_tokens=tokenizer.batch_decode(trimmed_window),
            )
        )

    return examples


class SimilaritySearch:
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
        max_ctx_idx = self.ctx_locations[:, 0].max()

        # Reshape strides and strides mask
        mask = tokens != self.subject_tokenizer.pad_token_id
        strides_mask = mask.unfold(dimension=1, size=ctx_len, step=ctx_len)
        strides = tokens.unfold(dimension=1, size=ctx_len, step=ctx_len)

        # Filter for valid strides
        valid_strides_mask = strides_mask.all(dim=2)
        valid_strides = strides[valid_strides_mask]

        # Embed context windows
        str_data = self.subject_tokenizer.batch_decode(valid_strides)
        print("Encoding token contexts...")
        embeddings = self.model.encode(
            str_data,
            device="cuda",
            show_progress_bar=True,
            convert_to_tensor=True,
        )

        # Set all strides and valid strides
        self.all_strides = strides
        self.valid_strides = valid_strides

        # Normalize embeddings
        normalized_embeddings = (
            embeddings / embeddings.norm(dim=1, keepdim=True)
        ).t()
        self.normalized_embeddings = normalized_embeddings.to("cuda")

    def _query(
        self,
        features: List[Feature],
        query_embeddings: TensorType["n_queries", "d_model"],
        k: int = 10,
    ):
        query_embeddings_normalized = query_embeddings / query_embeddings.norm(
            dim=1, keepdim=True
        )

        similarities: TensorType["n_queries", "n_contexts"] = t.matmul(
            query_embeddings_normalized, self.normalized_embeddings
        )

        # # set similarities to -inf for the stride indices
        for i, features in enumerate(features):
            idx = features.index
            mask = self.ctx_locations[:, 1] == idx
            locations = self.ctx_locations[mask]
            similarities[i, locations[:, 0]] = -float("inf")

        print(self.ctx_locations.shape)
        print(similarities)

        _, indices = t.topk(similarities, k=k, dim=1)

        return indices.cpu()

    def _get_similar_examples(
        self,
        idxs: TensorType["k"],
    ):
        print(idxs)
        examples = []
        for idx in idxs:
            token_window = self.all_strides[idx]
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
                    quantile=-1,
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
    ):
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
            topk_indices = self._query(features, query_batch, k=n_examples)
            for idxs, feature in zip(topk_indices, features):
                feature.similar_examples = self._get_similar_examples(idxs)


def default_sampler(
    token_windows: TensorType["batch", "seq"],
    activation_windows: TensorType["batch", "seq"],
    tokenizer: AutoTokenizer,
    n_train: int = 20,
    n_test: int = 40,
    n_quantiles: int = 10,
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
