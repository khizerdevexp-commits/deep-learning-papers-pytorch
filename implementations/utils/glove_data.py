"""Text loading and co-occurrence datasets for GloVe."""

from collections import Counter
from pathlib import Path
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
from torch.utils.data import DataLoader, Dataset

from implementations.models.glove import build_cooccurrence_matrix


UNKNOWN_TOKEN = "<unk>"
# Match words with optional internal apostrophes, or individual punctuation;
# \\w is Unicode-aware here, and \\s whitespace is excluded from punctuation.
_TOKEN_PATTERN = re.compile(r"\w+(?:['’]\w+)*|[^\w\s]", flags=re.UNICODE)


def tokenize(text: str, lowercase: bool = True) -> List[str]:
    """Split a text line into words and punctuation tokens."""
    if lowercase:
        text = text.lower()
    return _TOKEN_PATTERN.findall(text)


def build_vocabulary(
    corpus: Iterable[Sequence[str]],
    min_count: int = 1,
    max_vocab_size: Optional[int] = None,
) -> Dict[str, int]:
    """Build a deterministic token-to-id mapping from tokenized sentences.

    ID 0 is reserved for ``<unk>``. If ``max_vocab_size`` is supplied, it
    includes this reserved entry.
    """
    if min_count <= 0:
        raise ValueError("min_count must be positive")
    if max_vocab_size is not None and max_vocab_size <= 0:
        raise ValueError("max_vocab_size must be positive")
    # the for clauses are evaluated left to right,it visits each sentence, then each token in that sentence
    counts = Counter(token for sentence in corpus for token in sentence)
    words = [
        (word, count)
        for word, count in counts.items()
        if word != UNKNOWN_TOKEN and count >= min_count
    ]
    # -item[1] is count and sorts by count descending, item[0] is word and sorts by word ascending
    words.sort(key=lambda item: (-item[1], item[0]))
    if max_vocab_size is not None:
        words = words[: max_vocab_size - 1]
    
    vocabulary = {UNKNOWN_TOKEN: 0}
    # enumearte wraps each item with its index, so the loop receives (index, (word, count)).
    vocabulary.update({word: index for index, (word, _) in enumerate(words, start=1)})
    return vocabulary


def encode_corpus(
    corpus: Iterable[Sequence[str]],
    vocabulary: Dict[str, int],
) -> List[List[int]]:
    """Convert tokenized sentences to vocabulary IDs; map missing tokens to UNK."""
    unknown_id = vocabulary[UNKNOWN_TOKEN]
    return [
        [vocabulary.get(token, unknown_id) for token in sentence]
        for sentence in corpus
    ]


class CooccurrenceDataset(Dataset):
    """Dataset of observed GloVe ``(target, context, count)`` pairs.

    Args:
        corpus: Sentences represented as vocabulary IDs.
        vocab_size: Number of vocabulary entries, including any reserved IDs.
        window_size: Maximum context distance on either side of each target.
    """

    def __init__(
        self,
        corpus: Iterable[Sequence[int]],
        vocab_size: int,
        window_size: int = 10,
    ) -> None:
        self.token_ids = [list(sentence) for sentence in corpus]
        self.cooccurrence_matrix = build_cooccurrence_matrix(
            self.token_ids,
            vocab_size=vocab_size,
            window_size=window_size,
        )
        self.pair_ids = self.cooccurrence_matrix.indices()
        self.counts = self.cooccurrence_matrix.values()

    def __len__(self) -> int:
        return self.counts.numel()

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.pair_ids[0, index], self.pair_ids[1, index], self.counts[index]


def load_glove_data(
    file_path: Union[str, Path],
    window_size: int = 10,
    batch_size: int = 512,
    min_count: int = 1,
    max_vocab_size: Optional[int] = None,
    lowercase: bool = True,
    shuffle: bool = True,
    num_workers: int = 0,
    encoding: str = "utf-8",
) -> Tuple[DataLoader, Dict[str, int]]:
    """Load a line-oriented text corpus as a GloVe co-occurrence DataLoader.

    Each nonempty input line is treated as a sentence so context windows do
    not cross line boundaries. The returned batches contain target IDs,
    context IDs, and inverse-distance-weighted positive co-occurrence counts.
    The underlying dataset exposes ``cooccurrence_matrix`` for direct use
    with ``GloveModel.fit``.

    Returns:
        A pair ``(data_loader, vocabulary)`` where vocabulary maps tokens to
        integer IDs and reserves ID 0 for ``<unk>``.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative")

    path = Path(file_path)
    tokenized_corpus = []
    with path.open("r", encoding=encoding) as corpus_file:
        for line in corpus_file:
            sentence = tokenize(line, lowercase=lowercase)
            if sentence:
                tokenized_corpus.append(sentence)

    if not tokenized_corpus:
        raise ValueError(f"No tokens found in corpus file: {path}")

    vocabulary = build_vocabulary(
        tokenized_corpus,
        min_count=min_count,
        max_vocab_size=max_vocab_size,
    )
    token_ids = encode_corpus(tokenized_corpus, vocabulary)
    dataset = CooccurrenceDataset(
        token_ids,
        vocab_size=len(vocabulary),
        window_size=window_size,
    )
    if len(dataset) == 0:
        raise ValueError("Corpus has no word-context pairs; add a line with at least two tokens")

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
    return data_loader, vocabulary
