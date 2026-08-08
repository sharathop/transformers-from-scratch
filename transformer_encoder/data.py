"""Dataset loaders, vocabulary construction, and tensorized text datasets."""
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

import torch
from datasets import load_dataset  # Hugging Face Datasets: downloads/caches AG News.
from torch.utils.data import Dataset


EXAMPLES = [
    ("i loved this movie it was warm funny and brilliant", 1),
    ("the acting was wonderful and the story felt sincere", 1),
    ("an excellent film with a beautiful ending", 1),
    ("smart writing made every scene enjoyable", 1),
    ("the soundtrack and performances were fantastic", 1),
    ("a charming entertaining and surprisingly moving movie", 1),
    ("i would happily watch this delightful film again", 1),
    ("the director created a compelling emotional story", 1),
    ("great characters and a satisfying plot", 1),
    ("this was a joyful clever little film", 1),
    ("the movie was boring predictable and far too long", 0),
    ("i disliked the wooden acting and messy plot", 0),
    ("a terrible film with an awful ending", 0),
    ("the script was weak and the jokes were not funny", 0),
    ("every scene felt dull and unnecessary", 0),
    ("the disappointing soundtrack made it even worse", 0),
    ("i would not recommend this confusing movie", 0),
    ("the director wasted a promising idea", 0),
    ("bad characters and an unsatisfying story", 0),
    ("this was a joyless frustrating little film", 0),
]


class Vocabulary:
    """Whitespace tokenizer with PAD and UNK special tokens."""
    PAD, UNK = "<pad>", "<unk>"

    def __init__(self, texts: Iterable[str], max_size: int = 30_000, min_frequency: int = 2):
        counts = Counter(word for text in texts for word in text.lower().split())
        words = [word for word, count in counts.most_common(max_size) if count >= min_frequency]
        self.itos = [self.PAD, self.UNK] + words
        self.stoi = {word: index for index, word in enumerate(self.itos)}

    def encode(self, text: str, max_length: int) -> list[int]:
        ids = [self.stoi.get(word, self.stoi[self.UNK]) for word in text.lower().split()]
        return (ids[:max_length] + [self.stoi[self.PAD]] * max_length)[:max_length]

    def __len__(self) -> int:
        return len(self.itos)


@dataclass
class TextDataset(Dataset):
    examples: list[tuple[str, int]]
    vocab: Vocabulary
    max_length: int = 14

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        text, label = self.examples[index]
        return torch.tensor(self.vocab.encode(text, self.max_length)), torch.tensor(label), text


def make_toy_splits(train_fraction: float = 0.7, max_length: int = 14):
    """Small offline fallback for a quick smoke test."""
    positive = [item for item in EXAMPLES if item[1] == 1]
    negative = [item for item in EXAMPLES if item[1] == 0]
    cutoff = int(len(positive) * train_fraction)
    train = positive[:cutoff] + negative[:cutoff]
    test = positive[cutoff:] + negative[cutoff:]
    vocab = Vocabulary((text for text, _ in train), min_frequency=1)
    return TextDataset(train, vocab, max_length), TextDataset(test, vocab, max_length), vocab, 2, ["negative", "positive"]


def make_ag_news_splits(train_samples: int = 30_000, test_samples: int = 7_600, max_length: int = 64, seed: int = 7):
    """Download AG News and expose a shuffled 30k training subset plus official test data.

    The Hugging Face ``datasets`` cache stores the download locally after first use.
    """
    if train_samples > 120_000:
        raise ValueError("AG News has 120,000 official training examples")
    if test_samples > 7_600:
        raise ValueError("AG News has 7,600 official test examples")
    dataset = load_dataset("ag_news")
    train_rows = dataset["train"].shuffle(seed=seed).select(range(train_samples))
    test_rows = dataset["test"].select(range(test_samples))
    train = list(zip(train_rows["text"], train_rows["label"]))
    test = list(zip(test_rows["text"], test_rows["label"]))
    vocab = Vocabulary((text for text, _ in train))
    labels = ["World", "Sports", "Business", "Sci/Tech"]
    return TextDataset(train, vocab, max_length), TextDataset(test, vocab, max_length), vocab, 4, labels


def make_splits(dataset_name: str = "ag_news", **kwargs):
    if dataset_name == "ag_news":
        return make_ag_news_splits(**kwargs)
    if dataset_name == "toy":
        return make_toy_splits(**kwargs)
    raise ValueError(f"Unknown dataset: {dataset_name}")
