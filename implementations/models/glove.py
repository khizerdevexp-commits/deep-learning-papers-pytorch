"""PyTorch implementation of GloVe (Global Vectors for Word Representation).

Reference: Pennington, Socher, and Manning (2014),
"GloVe: Global Vectors for Word Representation".
https://nlp.stanford.edu/projects/glove/

For every observed word-context pair (i, j), the model minimizes the paper's
weighted least-squares objective (Eq. 8):

    J = sum_ij f(X_ij) * (w_i.T @ w_tilde_j + b_i + b_tilde_j
                          - log(X_ij)) ** 2

where f(x) = (x / x_max) ** alpha for x < x_max, and 1 otherwise (Eq. 9).
"""

from typing import Iterable, List, Sequence, Tuple, Union

import torch
from torch import nn


def build_cooccurrence_matrix(
    corpus: Iterable[Sequence[int]],
    vocab_size: int,
    window_size: int = 10,
) -> torch.Tensor:
    """Build a sparse, distance-weighted word-context matrix.

    Each token is a target once, and tokens on both sides within the window
    are contexts. A context at distance d contributes 1/d, as in the GloVe
    paper's corpus construction. The result is a coalesced sparse COO tensor.

    Args:
        corpus: Iterable of token-id sequences, one sequence per sentence or
            document. Sentence boundaries are not crossed.
        vocab_size: Number of entries in the vocabulary.
        window_size: Maximum context distance on either side of a target.
    """
    if vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    if window_size <= 0:
        raise ValueError("window_size must be positive")

    cooccurrences = {}
    for sequence in corpus:
        token_ids = [int(token_id) for token_id in sequence]
        for target_position, target_id in enumerate(token_ids):
            if not 0 <= target_id < vocab_size:
                raise ValueError(f"token id {target_id} is outside the vocabulary")

            start = max(0, target_position - window_size)
            end = min(len(token_ids), target_position + window_size + 1)
            for context_position in range(start, end):
                if context_position == target_position:
                    continue

                context_id = token_ids[context_position]
                if not 0 <= context_id < vocab_size:
                    raise ValueError(
                        f"token id {context_id} is outside the vocabulary"
                    )

                # Eq. (1) uses word-context counts; the paper weights each
                # occurrence inversely by its distance from the target.
                distance = abs(target_position - context_position)
                pair = (target_id, context_id)
                cooccurrences[pair] = cooccurrences.get(pair, 0.0) + 1.0 / distance

    if not cooccurrences:
        indices = torch.empty((2, 0), dtype=torch.long)
        values = torch.empty((0,), dtype=torch.float32)
    else:
        indices = torch.tensor(list(cooccurrences), dtype=torch.long).t().contiguous()
        values = torch.tensor(list(cooccurrences.values()), dtype=torch.float32)

    return torch.sparse_coo_tensor(
        indices, values, size=(vocab_size, vocab_size)
    ).coalesce()


def glove_loss(
    predictions: torch.Tensor,
    counts: torch.Tensor,
    x_max: float = 100.0,
    alpha: float = 0.75,
    reduction: str = "sum",
) -> torch.Tensor:
    """Compute the GloVe weighted least-squares objective (Eqs. 8-9).

    ``counts`` must contain only positive co-occurrence values. ``sum`` is
    the literal paper objective; ``mean`` can be useful for larger batches.
    """
    if x_max <= 0:
        raise ValueError("x_max must be positive")
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    if reduction not in {"sum", "mean"}:
        raise ValueError("reduction must be 'sum' or 'mean'")
    if predictions.shape != counts.shape:
        raise ValueError("predictions and counts must have the same shape")
    if torch.any(counts <= 0):
        raise ValueError("GloVe loss requires strictly positive co-occurrence counts")

    counts = counts.to(device=predictions.device, dtype=predictions.dtype)
    weights = (counts / x_max).clamp(max=1.0).pow(alpha)
    # This is f(X_ij) * (prediction - log(X_ij))^2.
    losses = weights * (predictions - counts.log()).square()
    return losses.sum() if reduction == "sum" else losses.mean()


class GloveModel(nn.Module):
    """GloVe embeddings with independent target/context vectors and biases.

    Args:
        vocab_size: Number of vocabulary entries.
        embedding_dim: Dimension d of each word and context vector.
        x_max: Count threshold in the paper's weighting function.
        alpha: Exponent in the paper's weighting function.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        x_max: float = 100.0,
        alpha: float = 0.75,
    ) -> None:
        super().__init__()
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if x_max <= 0:
            raise ValueError("x_max must be positive")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.x_max = x_max
        self.alpha = alpha

        # GloVe has distinct word and context vectors and one bias per role.
        self.target_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.target_biases = nn.Embedding(vocab_size, 1)
        self.context_biases = nn.Embedding(vocab_size, 1)

        # Small random vectors and zero biases provide a simple stable start.
        bound = 0.5 / embedding_dim
        nn.init.uniform_(self.target_embeddings.weight, -bound, bound)
        nn.init.uniform_(self.context_embeddings.weight, -bound, bound)
        nn.init.zeros_(self.target_biases.weight)
        nn.init.zeros_(self.context_biases.weight)

    def forward(
        self,
        target_ids: torch.Tensor,
        context_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Return predicted log co-occurrence for each pair.

        Implements w_i.T @ w_tilde_j + b_i + b_tilde_j. The ID tensors can
        have any matching shape; the returned tensor has that same shape.
        """
        target_vectors = self.target_embeddings(target_ids)
        context_vectors = self.context_embeddings(context_ids)
        dot_products = (target_vectors * context_vectors).sum(dim=-1)
        target_bias = self.target_biases(target_ids).squeeze(-1)
        context_bias = self.context_biases(context_ids).squeeze(-1)
        return dot_products + target_bias + context_bias

    def loss(
        self,
        target_ids: torch.Tensor,
        context_ids: torch.Tensor,
        counts: torch.Tensor,
        reduction: str = "sum",
    ) -> torch.Tensor:
        """Compute Eq. (8) for a batch of observed co-occurrence pairs."""
        predictions = self(target_ids, context_ids)
        return glove_loss(
            predictions,
            counts,
            x_max=self.x_max,
            alpha=self.alpha,
            reduction=reduction,
        )

    def fit(
        self,
        cooccurrence: torch.Tensor,
        epochs: int = 25,
        learning_rate: float = 0.05,
        batch_size: int = 1,
        shuffle: bool = True,
    ) -> List[float]:
        """Train on a sparse co-occurrence matrix using AdaGrad.

        The paper uses AdaGrad updates. A batch size of one is the closest
        match to its per-example stochastic updates; larger batches are
        available for faster training. Returns the summed objective per epoch.

        Args:
            cooccurrence: Square sparse COO matrix from
                :func:`build_cooccurrence_matrix`, or another square count
                matrix containing nonnegative values.
            epochs: Number of passes over observed pairs.
            learning_rate: AdaGrad learning rate (paper default: 0.05).
            batch_size: Number of pairs per optimizer update (default: 1).
            shuffle: Shuffle observed pairs at the start of every epoch.
        """
        if epochs <= 0:
            raise ValueError("epochs must be positive")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if cooccurrence.ndim != 2 or tuple(cooccurrence.shape) != (
            self.vocab_size,
            self.vocab_size,
        ):
            raise ValueError("cooccurrence must have shape (vocab_size, vocab_size)")

        sparse_counts = cooccurrence.to_sparse_coo().coalesce()
        pair_ids = sparse_counts.indices()
        counts = sparse_counts.values()
        if counts.numel() == 0:
            raise ValueError("cooccurrence matrix has no observed pairs")
        if torch.any(counts <= 0):
            raise ValueError("cooccurrence matrix must contain only positive stored counts")

        device = self.target_embeddings.weight.device
        pair_ids = pair_ids.to(device=device)
        counts = counts.to(
            device=device,
            dtype=self.target_embeddings.weight.dtype,
        )
        optimizer = torch.optim.Adagrad(self.parameters(), lr=learning_rate)
        num_pairs = counts.numel()
        history = []

        self.train()
        for _ in range(epochs):
            if shuffle:
                order = torch.randperm(num_pairs, device=device)
            else:
                order = torch.arange(num_pairs, device=device)

            epoch_loss = 0.0
            for start in range(0, num_pairs, batch_size):
                batch = order[start : start + batch_size]
                target_ids = pair_ids[0, batch]
                context_ids = pair_ids[1, batch]
                batch_counts = counts[batch]

                optimizer.zero_grad(set_to_none=True)
                batch_loss = self.loss(
                    target_ids, context_ids, batch_counts, reduction="sum"
                )
                batch_loss.backward()
                optimizer.step()
                epoch_loss += batch_loss.detach().item()

            history.append(epoch_loss)

        return history

    def get_embeddings(
        self,
        combine: bool = True,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Return word vectors, optionally combining both learned roles.

        GloVe commonly uses w_i + w_tilde_i as the final word vector. Set
        ``combine=False`` to access the target and context tables separately.
        """
        target = self.target_embeddings.weight
        context = self.context_embeddings.weight
        return target + context if combine else (target, context)
